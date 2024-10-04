// Copyright (C) Microsoft Corporation. All rights reserved.

//! This module contains Underhill specific functionality and implementations of require traits
//! in order to plug into the rest of the common HvLite code.

pub mod mshv;
mod nice;
mod vp_state;

cfg_if::cfg_if! {
    if #[cfg(guest_arch = "x86_64")] {
        mod hardware_cvm;
        pub mod snp;
        pub mod tdx;

        use crate::VtlCrash;
        use hardware_cvm::GuestVsmVpState;
        use hvdef::HvX64RegisterName;
        use hvdef::HvX64SegmentRegister;
        use virt::x86::MsrError;
        use virt::vp::AccessVpState;
        #[cfg(feature = "gdb")]
        use virt::x86::HardwareBreakpoint;
    } else if #[cfg(guest_arch = "aarch64")] {
        use hv1_hypercall::Arm64RegisterState;
        use hvdef::HvArm64RegisterName;
    } else {
        compile_error!("unsupported guest architecture");
    }
}

use super::Error;
use super::UhPartitionInner;
use super::UhVpInner;
use crate::BackingShared;
use crate::GuestVsmState;
use crate::WakeReason;
use guestmem::GuestMemory;
use hcl::ioctl;
use hcl::ioctl::ProcessorRunner;
use hv1_emulator::hv::ProcessorVtlHv;
use hv1_emulator::message_queues::MessageQueues;
use hv1_emulator::synic::ProcessorSynic;
use hvdef::hypercall::HostVisibilityType;
use hvdef::HvError;
use hvdef::HvMessage;
use hvdef::HvSynicSint;
use hvdef::Vtl;
use hvdef::NUM_SINTS;
use inspect::Inspect;
use inspect::InspectMut;
use pal::unix::affinity;
use pal::unix::affinity::CpuSet;
use pal_async::driver::Driver;
use pal_async::driver::PollImpl;
use pal_async::timer::PollTimer;
use pal_uring::IdleControl;
use parking_lot::Mutex;
use private::BackingPrivate;
use std::convert::Infallible;
use std::future::poll_fn;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;
use virt::io::CpuIo;
use virt::Processor;
use virt::StopVp;
use virt::VpHaltReason;
use virt::VpIndex;
use vm_topology::processor::TargetVpInfo;
use vmcore::vmtime::VmTimeAccess;
use vtl_array::VtlArray;

/// An object to run lower VTLs and to access processor state.
///
/// This is not [`Send`] and can only be instantiated from
/// [`crate::UhProcessorBox::bind_processor`]. This ensures that it can only be used
/// from a thread that is affinitized to the VP, since it is only possible to
/// access lower VTL processor state from the same processor.
#[derive(InspectMut)]
#[inspect(extra = "UhProcessor::inspect_extra", bound = "T: Backing")]
pub struct UhProcessor<'a, T: Backing> {
    _not_send: PhantomData<*mut ()>,

    #[inspect(flatten)]
    inner: &'a UhVpInner,
    #[inspect(skip)]
    partition: &'a UhPartitionInner,
    #[inspect(skip)]
    idle_control: Option<&'a mut IdleControl>,
    #[inspect(skip)]
    kernel_returns: u64,
    #[inspect(with = "|x| inspect::iter_by_index(x).map_value(inspect::AsHex)")]
    crash_reg: [u64; hvdef::HV_X64_GUEST_CRASH_PARAMETER_MSRS],
    #[inspect(with = "|x| inspect::AsHex(u64::from(*x))")]
    crash_control: hvdef::GuestCrashCtl,
    hv: VtlArray<Option<ProcessorVtlHv>, 2>,
    untrusted_synic: Option<ProcessorSynic>,
    vmtime: VmTimeAccess,
    #[inspect(skip)]
    timer: PollImpl<dyn PollTimer>,
    #[inspect(mut)]
    force_exit_sidecar: bool,
    /// The VTLs on this VP that are currently locked, per requesting VTL.
    vtls_tlb_locked: VtlsTlbLocked,
    /// The VTLs on this VP waiting for TLB locks on other VPs.
    // Only used on HCVM.
    vtls_tlb_waiting: VtlArray<bool, 2>,
    #[cfg(guest_arch = "x86_64")]
    cvm_guest_vsm: Option<GuestVsmVpState>,

    // Put the runner and backing at the end so that monomorphisms of functions
    // that don't access backing-specific state are more likely to be folded
    // together by the compiler.
    #[inspect(skip)]
    runner: ProcessorRunner<'a, T::HclBacking>,
    #[inspect(mut)]
    backing: T,
}

#[derive(Inspect)]
struct VtlsTlbLocked {
    // vtl0: VtlArray<bool, 0>,
    vtl1: VtlArray<bool, 1>,
    vtl2: VtlArray<bool, 2>,
}

#[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
impl VtlsTlbLocked {
    fn get(&self, requesting_vtl: Vtl, target_vtl: Vtl) -> bool {
        match requesting_vtl {
            Vtl::Vtl0 => unreachable!(),
            Vtl::Vtl1 => self.vtl1[target_vtl],
            Vtl::Vtl2 => self.vtl2[target_vtl],
        }
    }

    fn set(&mut self, requesting_vtl: Vtl, target_vtl: Vtl, value: bool) {
        match requesting_vtl {
            Vtl::Vtl0 => unreachable!(),
            Vtl::Vtl1 => self.vtl1[target_vtl] = value,
            Vtl::Vtl2 => self.vtl2[target_vtl] = value,
        }
    }

    fn fill(&mut self, requesting_vtl: Vtl, value: bool) {
        match requesting_vtl {
            Vtl::Vtl0 => unreachable!(),
            Vtl::Vtl1 => self.vtl1.fill(value),
            Vtl::Vtl2 => self.vtl2.fill(value),
        }
    }
}

mod private {
    use super::vp_state;
    use super::UhRunVpError;
    use crate::processor::UhProcessor;
    use crate::BackingShared;
    use crate::Error;
    use crate::UhPartitionInner;
    use hcl::ioctl::ProcessorRunner;
    use hvdef::Vtl;
    use inspect::InspectMut;
    use std::future::Future;
    use virt::io::CpuIo;
    use virt::vp::AccessVpState;
    use virt::StopVp;
    use virt::VpHaltReason;
    use vm_topology::processor::TargetVpInfo;

    pub struct BackingParams<'a, 'b, T: BackingPrivate> {
        pub(crate) partition: &'a UhPartitionInner,
        pub(crate) vp_info: &'a TargetVpInfo,
        pub(crate) runner: &'a mut ProcessorRunner<'b, T::HclBacking>,
        pub(crate) backing_shared: &'a BackingShared,
    }

    pub trait BackingPrivate: 'static + Sized + InspectMut + Sized {
        type BackingShared;

        type HclBacking: hcl::ioctl::Backing;

        fn new_shared_state(
            params: super::BackingSharedParams<'_>,
        ) -> Result<Self::BackingShared, Error>;

        fn new(params: BackingParams<'_, '_, Self>) -> Result<Self, Error>;

        type StateAccess<'p, 'a>: AccessVpState<Error = vp_state::Error>
        where
            Self: 'a + 'p,
            'p: 'a;

        fn init(this: &mut UhProcessor<'_, Self>);

        fn access_vp_state<'a, 'p>(
            this: &'a mut UhProcessor<'p, Self>,
            vtl: Vtl,
        ) -> Self::StateAccess<'p, 'a>;

        fn run_vp(
            this: &mut UhProcessor<'_, Self>,
            dev: &impl CpuIo,
            stop: &mut StopVp<'_>,
        ) -> impl Future<Output = Result<(), VpHaltReason<UhRunVpError>>>;

        /// Returns true if the VP is ready to run, false if it is halted.
        fn poll_apic(
            this: &mut UhProcessor<'_, Self>,
            scan_irr: bool,
        ) -> Result<bool, UhRunVpError>;

        /// Requests the VP to exit when an external interrupt is ready to be
        /// delivered.
        ///
        /// Only used when the hypervisor implements the APIC.
        fn request_extint_readiness(this: &mut UhProcessor<'_, Self>);

        /// Requests the VP to exit when any of the specified SINTs have a free
        /// message slot.
        ///
        /// This is used for hypervisor-managed and untrusted SINTs.
        fn request_untrusted_sint_readiness(this: &mut UhProcessor<'_, Self>, sints: u16);

        /// The VTL that was running when the VP exited into VTL2, with the
        /// exception of a successful vtl switch, where it will return the VTL
        /// that will run on VTL 2 exit.
        fn last_vtl(this: &UhProcessor<'_, Self>) -> Vtl;

        /// Copies shared registers (per VSM TLFS spec) from the last VTL to
        /// the target VTL that will become active.
        fn switch_vtl_state(this: &mut UhProcessor<'_, Self>, target_vtl: Vtl);
    }
}

pub struct BackingSharedParams<'a> {
    #[cfg(guest_arch = "x86_64")]
    pub(crate) cvm_state: Option<&'a crate::UhCvmPartitionState>,
    #[cfg(not(guest_arch = "x86_64"))]
    pub(crate) _phantom: &'a (),
}

/// Processor backing.
pub trait Backing: BackingPrivate {
    /// Construct a state object that should be shared amongst all VPs in the partition of this Backing.
    fn new_shared_state(params: BackingSharedParams<'_>) -> Result<Self::BackingShared, Error> {
        <Self as BackingPrivate>::new_shared_state(params)
    }
}

impl<T: BackingPrivate> Backing for T {}

/// Marker trait for processor backings that have hardware isolation support.
pub trait HardwareIsolatedBacking: Backing {}

#[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
#[derive(Inspect, Debug)]
#[inspect(tag = "reason")]
pub enum SidecarExitReason {
    #[inspect(transparent)]
    Exit(SidecarRemoveExit),
    #[inspect(transparent)]
    TaskRequest(Arc<str>),
    ManualRequest,
}

#[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
#[derive(Inspect, Debug)]
#[inspect(tag = "exit")]
pub enum SidecarRemoveExit {
    Msr {
        #[inspect(hex)]
        msr: u32,
        value: Option<u64>,
    },
    Io {
        #[inspect(hex)]
        port: u16,
        write: bool,
    },
    Mmio {
        #[inspect(hex)]
        gpa: u64,
        write: bool,
    },
    Hypercall {
        #[inspect(debug)]
        code: hvdef::HypercallCode,
    },
    Cpuid {
        #[inspect(hex)]
        leaf: u32,
        #[inspect(hex)]
        subleaf: u32,
    },
    Hypervisor {
        #[inspect(debug)]
        message: hvdef::HvMessageType,
    },
}

impl UhVpInner {
    // Create a new vp's state.
    pub fn new(cpu_index: u32, vp_info: TargetVpInfo, vp_count: usize) -> Self {
        Self {
            wake_reasons: Default::default(),
            message_queues: VtlArray::from_fn(|_| MessageQueues::new()),
            waker: Default::default(),
            cpu_index,
            vp_info,
            vtl1_enabled: Mutex::new(false),
            hv_start_enable_vtl_vp: VtlArray::from_fn(|_| Mutex::new(None)),
            sidecar_exit_reason: Default::default(),
            tlb_lock_info: VtlArray::<_, 2>::from_fn(|_| super::TlbLockInfo::new(vp_count)),
        }
    }

    /// Queues a message for sending, optionally alerting the hypervisor if the queue is empty.
    pub fn post_message(&self, vtl: Vtl, sint: u8, message: &HvMessage) {
        if self.message_queues[vtl].enqueue_message(sint, message) {
            self.wake(vtl, WakeReason::MESSAGE_QUEUES);
        }
    }

    pub fn wake(&self, vtl: Vtl, reason: WakeReason) {
        let reason = u64::from(reason.0) << (vtl as u8 * 32);
        if self.wake_reasons.fetch_or(reason, Ordering::Release) & reason == 0 {
            if let Some(waker) = &*self.waker.read() {
                waker.wake_by_ref();
            }
        }
    }

    #[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
    pub fn set_sidecar_exit_reason(&self, reason: SidecarExitReason) {
        self.sidecar_exit_reason.lock().get_or_insert_with(|| {
            tracing::info!(?reason, "sidecar exit");
            reason
        });
    }
}

/// Underhill-specific run VP error
#[derive(Debug, Error)]
pub enum UhRunVpError {
    /// Failed to run
    #[error("failed to run")]
    Run(#[source] ioctl::Error),
    #[error("sidecar run error")]
    Sidecar(#[source] sidecar_client::SidecarError),
    /// Failed to access state for emulation
    #[error("failed to access state for emulation")]
    EmulationState(#[source] ioctl::Error),
    /// Failed to access state for hypercall handling
    #[error("failed to access state for hypercall handling")]
    HypercallState(#[source] ioctl::Error),
    /// Failed to translate GVA
    #[error("failed to translate GVA")]
    TranslateGva(#[source] ioctl::Error),
    /// Failed VTL access check
    #[error("failed VTL access check")]
    VtlAccess(#[source] ioctl::Error),
    /// Failed to advance rip
    #[error("failed to advance rip")]
    AdvanceRip(#[source] ioctl::Error),
    /// Failed to set pending event
    #[error("failed to set pending event")]
    Event(#[source] ioctl::Error),
    /// Guest accessed unaccepted gpa
    #[error("guest accessed unaccepted gpa {0}")]
    UnacceptedMemoryAccess(u64),
    /// State access error
    #[error("state access error")]
    State(#[source] vp_state::Error),
    /// Invalid vmcb
    #[error("invalid vmcb")]
    InvalidVmcb,
    #[error("unknown exit {0:#x?}")]
    UnknownVmxExit(x86defs::vmx::VmxExit),
    /// Failed to read hypercall parameters
    #[error("failed to read hypercall parameters")]
    HypercallParameters(#[source] guestmem::GuestMemoryError),
    /// Failed to write hypercall result
    #[error("failed to write hypercall result")]
    HypercallResult(#[source] guestmem::GuestMemoryError),
    #[error("failed to write hypercall control for retry")]
    HypercallRetry(#[source] guestmem::GuestMemoryError),
    #[error("unexpected debug exception with dr6 value {0:#x}")]
    UnexpectedDebugException(u64),
}

/// Underhill processor run error
#[derive(Debug, Error)]
pub enum ProcessorError {
    /// IOCTL error
    #[error("hcl error")]
    Ioctl(#[from] ioctl::Error),
    /// State access error
    #[error("state access error")]
    State(#[from] vp_state::Error),
    /// Not supported
    #[error("operation not supported")]
    NotSupported,
}

fn duration_from_100ns(n: u64) -> Duration {
    const NUM_100NS_IN_SEC: u64 = 10 * 1000 * 1000;
    Duration::new(n / NUM_100NS_IN_SEC, (n % NUM_100NS_IN_SEC) as u32 * 100)
}

impl<'a, T: Backing> UhProcessor<'a, T> {
    fn inspect_extra(&mut self, resp: &mut inspect::Response<'_>) {
        resp.child("stats", |req| {
            // Get all the VP stats and just grab this VP's.
            if let Ok(stats) = hcl::stats::vp_stats() {
                let stats = &stats[self.vp_index().index() as usize];
                req.respond()
                    .counter("vtl_transitions", stats.vtl_transitions)
                    .counter(
                        "spurious_exits",
                        stats.vtl_transitions.saturating_sub(self.kernel_returns),
                    );
            }
        })
        .field(
            "last_enter_modes",
            self.runner
                .enter_mode()
                .map(|&mut v| inspect::AsHex(u8::from(v))),
        )
        .field("sidecar", self.runner.is_sidecar())
        .field(
            "sidecar_base_cpu",
            self.partition.hcl.sidecar_base_cpu(self.vp_index().index()),
        );
    }

    fn update_synic(&mut self, vtl: Vtl, untrusted_synic: bool) {
        loop {
            let hv = self.hv[vtl].as_mut().unwrap();

            let ref_time_now = hv.ref_time_now();
            let synic = if untrusted_synic {
                debug_assert_eq!(vtl, Vtl::Vtl0);
                self.untrusted_synic.as_mut().unwrap()
            } else {
                &mut hv.synic
            };
            let (ready_sints, next_ref_time) = synic.scan(
                ref_time_now,
                &self.partition.gm[vtl],
                &mut self
                    .partition
                    .synic_interrupt(self.inner.vp_info.base.vp_index, vtl),
            );
            if let Some(next_ref_time) = next_ref_time {
                // Convert from reference timer basis to vmtime basis via
                // difference of programmed timer and current reference time.
                let ref_diff = next_ref_time.saturating_sub(ref_time_now);
                let timeout = self
                    .vmtime
                    .now()
                    .wrapping_add(duration_from_100ns(ref_diff));
                self.vmtime.set_timeout_if_before(timeout);
            }
            if ready_sints == 0 {
                break;
            }
            self.deliver_synic_messages(vtl, ready_sints);
            // Loop around to process the synic again.
        }
    }

    #[cfg(all(feature = "gdb", guest_arch = "x86_64"))]
    fn handle_debug_exception(&mut self) -> Result<(), VpHaltReason<UhRunVpError>> {
        // FUTURE: Underhill does not yet support VTL1 so this is only tested with VTL0.
        if self.last_vtl() == Vtl::Vtl0 {
            let debug_regs: virt::x86::vp::DebugRegisters = self
                .access_state(Vtl::Vtl0)
                .debug_regs()
                .expect("register query should not fail");

            let dr = [
                debug_regs.dr0,
                debug_regs.dr1,
                debug_regs.dr2,
                debug_regs.dr3,
            ];

            if debug_regs.dr6 & x86defs::DR6_SINGLE_STEP != 0 {
                return Err(VpHaltReason::SingleStep);
            }

            // Last four bits of DR6 indicate which breakpoint was triggered.
            const BREAKPOINT_INDEX_OFFSET: usize = 4;
            let i = debug_regs.dr6.trailing_zeros() as usize;
            if i >= BREAKPOINT_INDEX_OFFSET {
                // Received a debug exception not triggered by a breakpoint or single step.
                return Err(VpHaltReason::InvalidVmState(
                    UhRunVpError::UnexpectedDebugException(debug_regs.dr6),
                ));
            }
            let bp = HardwareBreakpoint::from_dr7(debug_regs.dr7, dr[i], i);

            return Err(VpHaltReason::HwBreak(bp));
        }

        panic!("unexpected debug exception in VTL {:?}", self.last_vtl());
    }
}

impl<'p, T: Backing> Processor for UhProcessor<'p, T> {
    type Error = ProcessorError;
    type RunVpError = UhRunVpError;
    type StateAccess<'a> = T::StateAccess<'p, 'a> where Self: 'a;

    #[cfg(guest_arch = "aarch64")]
    fn set_debug_state(
        &mut self,
        _vtl: Vtl,
        _state: Option<&virt::x86::DebugState>,
    ) -> Result<(), Self::Error> {
        Err(ProcessorError::NotSupported)
    }

    #[cfg(guest_arch = "x86_64")]
    fn set_debug_state(
        &mut self,
        vtl: Vtl,
        state: Option<&virt::x86::DebugState>,
    ) -> Result<(), Self::Error> {
        // FUTURE: Underhill does not yet support VTL1 so this is only tested with VTL0.
        if vtl == Vtl::Vtl0 {
            let mut db: [u64; 4] = [0; 4];
            let mut rflags =
                x86defs::RFlags::from(self.access_state(Vtl::Vtl0).registers().unwrap().rflags);
            let mut dr7: u64 = 0;

            if let Some(state) = state {
                rflags.set_trap(state.single_step);
                for (i, bp) in state.breakpoints.iter().enumerate() {
                    if let Some(bp) = bp {
                        db[i] = bp.address;
                        dr7 |= bp.dr7_bits(i);
                    }
                }
            }

            let debug_registers = virt::x86::vp::DebugRegisters {
                dr0: db[0],
                dr1: db[1],
                dr2: db[2],
                dr3: db[3],
                dr6: 0,
                dr7,
            };

            let mut access_state = self.access_state(vtl);

            access_state.set_debug_regs(&debug_registers)?;

            let registers = {
                let mut registers = access_state.registers().unwrap();
                registers.rflags = rflags.into();
                registers
            };
            access_state.set_registers(&registers)?;

            return Ok(());
        }

        panic!("unexpected set debug state in VTL {:?}", vtl);
    }

    async fn run_vp(
        &mut self,
        mut stop: StopVp<'_>,
        dev: &impl CpuIo,
    ) -> Result<Infallible, VpHaltReason<UhRunVpError>> {
        if self.runner.is_sidecar() {
            if self.force_exit_sidecar {
                self.inner
                    .set_sidecar_exit_reason(SidecarExitReason::ManualRequest);
                return Err(VpHaltReason::Cancel);
            }
        } else {
            {
                let mut current = Default::default();
                affinity::get_current_thread_affinity(&mut current).unwrap();
                assert_eq!(&current, CpuSet::new().set(self.inner.cpu_index));
            }

            // Lower the priority of this VP thread so that the VM does not return
            // to VTL0 while there is still outstanding VTL2 work to do.
            nice::nice(1);
        }

        let mut last_waker = None;

        // Force deliverability notifications to be reevaluated.
        let vtl0_wakes = WakeReason::new()
            .with_message_queues(true)
            .with_intcon(true);
        let vtl1_wakes = WakeReason::new().with_message_queues(true);
        self.inner.wake_reasons.fetch_or(
            ((vtl1_wakes.0 as u64) << 32) | (vtl0_wakes.0 as u64),
            Ordering::Relaxed,
        );

        let mut scan_irr = true;

        loop {
            // Process VP activity and wait for the VP to be ready.
            poll_fn(|cx| loop {
                stop.check()?;

                // Clear the run VP cancel request.
                self.runner.clear_cancel();

                // Cancel any pending timer.
                self.vmtime.cancel_timeout();

                // Ensure the waker is set.
                if !last_waker
                    .as_ref()
                    .map_or(false, |waker| cx.waker().will_wake(waker))
                {
                    last_waker = Some(cx.waker().clone());
                    self.inner.waker.write().clone_from(&last_waker);
                }

                // Process wakes.
                if self.inner.wake_reasons.load(Ordering::Relaxed) != 0 {
                    scan_irr = self.handle_wake().map_err(VpHaltReason::Hypervisor)?;
                }

                for vtl in [Vtl::Vtl1, Vtl::Vtl0] {
                    // Process interrupts.
                    if self.hv(vtl).is_some() {
                        self.update_synic(vtl, false);
                    }
                }
                if self.untrusted_synic.is_some() {
                    self.update_synic(Vtl::Vtl0, true);
                }

                let ready = T::poll_apic(self, scan_irr).map_err(VpHaltReason::Hypervisor)?;
                scan_irr = false;

                // Arm the timer.
                if let Some(timeout) = self.vmtime.get_timeout() {
                    let deadline = self.vmtime.host_time(timeout);
                    if self.timer.poll_timer(cx, deadline).is_ready() {
                        continue;
                    }
                }

                if ready {
                    return <Result<_, VpHaltReason<_>>>::Ok(()).into();
                }

                break Poll::Pending;
            })
            .await?;

            // Yield if the thread pool is not ready to block.
            if let Some(idle_control) = &mut self.idle_control {
                if !idle_control.pre_block() {
                    yield_now().await;
                    continue;
                }
            }

            if let Some(mode) = self.runner.enter_mode() {
                *mode = self
                    .partition
                    .enter_modes_atomic
                    .load(Ordering::Relaxed)
                    .into();
            }

            T::run_vp(self, dev, &mut stop).await?;
            self.kernel_returns += 1;
        }
    }

    fn flush_async_requests(&mut self) -> Result<(), Self::RunVpError> {
        if self.inner.wake_reasons.load(Ordering::Relaxed) != 0 {
            let scan_irr = self.handle_wake()?;
            if scan_irr {
                T::poll_apic(self, true)?;
            }
        }
        Ok(())
    }

    fn access_state(&mut self, vtl: Vtl) -> Self::StateAccess<'_> {
        T::access_vp_state(self, vtl)
    }
}

impl<'a, T: Backing> UhProcessor<'a, T> {
    pub(super) fn new(
        driver: &impl Driver,
        partition: &'a UhPartitionInner,
        vp_info: TargetVpInfo,
        backing_shared: &'a BackingShared,
        idle_control: Option<&'a mut IdleControl>,
    ) -> Result<Self, Error> {
        let inner = partition.vp(vp_info.base.vp_index).unwrap();
        let mut runner = partition
            .hcl
            .runner(inner.cpu_index, idle_control.is_none())
            .unwrap();
        let backing = T::new(private::BackingParams {
            partition,
            vp_info: &vp_info,
            runner: &mut runner,
            backing_shared,
        })?;

        let hv = {
            let vtl0_hv = partition.hv.as_ref().map(|hv| {
                hv.add_vp(
                    partition.gm[Vtl::Vtl0].clone(),
                    vp_info.base.vp_index,
                    Vtl::Vtl0,
                )
            });
            VtlArray::from([vtl0_hv, None])
        };

        let untrusted_synic = partition
            .untrusted_synic
            .as_ref()
            .map(|synic| synic.add_vp(vp_info.base.vp_index, Vtl::Vtl0));

        let mut vp = Self {
            partition,
            inner,
            runner,
            idle_control,
            kernel_returns: 0,
            crash_reg: [0; hvdef::HV_X64_GUEST_CRASH_PARAMETER_MSRS],
            crash_control: hvdef::GuestCrashCtl::new()
                .with_crash_notify(true)
                .with_crash_message(true),
            hv,
            untrusted_synic,
            _not_send: PhantomData,
            backing,
            vmtime: partition
                .vmtime
                .access(format!("vp-{}", vp_info.base.vp_index.index())),
            timer: driver.new_dyn_timer(),
            force_exit_sidecar: false,
            vtls_tlb_locked: VtlsTlbLocked {
                vtl1: VtlArray::new(false),
                vtl2: VtlArray::new(false),
            },
            vtls_tlb_waiting: VtlArray::<_, 2>::new(false),
            #[cfg(guest_arch = "x86_64")]
            cvm_guest_vsm: None,
        };

        T::init(&mut vp);

        Ok(vp)
    }

    /// Returns true if the interrupt controller has work to do.
    fn handle_wake(&mut self) -> Result<bool, UhRunVpError> {
        let wake_reasons_raw = self.inner.wake_reasons.swap(0, Ordering::SeqCst);
        let wake_reasons_vtl: [WakeReason; 2] = zerocopy::transmute!(wake_reasons_raw);
        for (vtl, wake_reasons) in [
            (Vtl::Vtl1, wake_reasons_vtl[1]),
            (Vtl::Vtl0, wake_reasons_vtl[0]),
        ] {
            if wake_reasons.message_queues() {
                let pending_sints = self.inner.message_queues[vtl].pending_sints();
                if pending_sints != 0 {
                    // Set SINT interest.
                    let pending_sints = self.inner.message_queues[vtl].pending_sints();
                    let mut masked_sints = 0;

                    // Determine which of the pending sints are masked.
                    for sint in 0..NUM_SINTS as u8 {
                        if pending_sints & (1 << sint) == 0 {
                            continue;
                        }
                        let sint_msr = if let Some(hv) = &self.hv(vtl) {
                            hv.synic.sint(sint)
                        } else {
                            #[cfg(guest_arch = "x86_64")]
                            let sint_reg =
                                HvX64RegisterName(HvX64RegisterName::Sint0.0 + sint as u32);
                            #[cfg(guest_arch = "aarch64")]
                            let sint_reg =
                                HvArm64RegisterName(HvArm64RegisterName::Sint0.0 + sint as u32);
                            self.runner.get_vp_register(sint_reg).unwrap().as_u64()
                        };
                        masked_sints |= (HvSynicSint::from(sint_msr).masked() as u16) << sint;
                    }

                    // Drain the queues for all masked SINTs.
                    self.inner.message_queues[vtl].post_pending_messages(masked_sints, |_, _| {
                        Err(HvError::InvalidSynicState)
                    });

                    self.request_sint_notifications(vtl, pending_sints & !masked_sints);
                }
            }

            if wake_reasons.extint() {
                T::request_extint_readiness(self);
            }

            #[cfg(guest_arch = "x86_64")]
            if wake_reasons.hv_start_enable_vtl_vp() {
                if let Some(context) = self.inner.hv_start_enable_vtl_vp[vtl].lock().take() {
                    tracing::debug!(
                        vp_index = self.inner.cpu_index,
                        ?vtl,
                        "starting vp with initial registers"
                    );
                    hv1_emulator::hypercall::set_x86_vp_context(
                        &mut self.access_state(vtl),
                        &context,
                    )
                    .map_err(UhRunVpError::State)?;

                    if vtl == Vtl::Vtl1 {
                        assert!(self.partition.is_hardware_isolated());
                        // Should not have already initialized the hv emulator for this vtl
                        assert!(self.hv(vtl).is_none());

                        // TODO GUEST_VSM construct VTL 1 lapics
                        self.hv[vtl] = Some(
                            self.partition
                                .hv
                                .as_ref()
                                .expect("has an hv emulator")
                                .add_vp(
                                    self.partition.gm[Vtl::Vtl1].clone(),
                                    self.vp_index(),
                                    Vtl::Vtl1,
                                ),
                        );
                        self.cvm_guest_vsm = Some(GuestVsmVpState {
                            // TODO CVM GUEST VSM: Revisit during AP startup if this is correct
                            current_vtl: Vtl::Vtl0,
                        })
                    }
                }
            }
        }

        Ok(wake_reasons_vtl[0].intcon())
    }

    fn request_sint_notifications(&mut self, vtl: Vtl, sints: u16) {
        if sints == 0 {
            return;
        }

        // Send the SINT notifications to the local synic for non-proxied SINTs.
        let untrusted_sints = if let Some(hv) = &mut self.hv[vtl] {
            let proxied_sints = hv.synic.proxied_sints();
            hv.synic.request_sint_readiness(sints & !proxied_sints);
            proxied_sints
        } else {
            !0
        };

        if sints & untrusted_sints != 0 {
            T::request_untrusted_sint_readiness(self, sints & untrusted_sints);
        }
    }

    fn vp_index(&self) -> VpIndex {
        self.inner.vp_info.base.vp_index
    }

    #[cfg(guest_arch = "x86_64")]
    fn write_msr(&mut self, msr: u32, value: u64) -> Result<(), MsrError> {
        let last_vtl = self.last_vtl();
        if msr & 0xf0000000 == 0x40000000 {
            if let Some(hv) = self.hv_mut(last_vtl) {
                let r = hv.msr_write(msr, value);
                if !matches!(r, Err(MsrError::Unknown)) {
                    return r;
                }
            }
        }
        match msr {
            hvdef::HV_X64_MSR_GUEST_CRASH_CTL => {
                self.crash_control = hvdef::GuestCrashCtl::from(value);
                let crash = VtlCrash {
                    vp_index: self.vp_index(),
                    last_vtl,
                    control: self.crash_control,
                    parameters: self.crash_reg,
                };
                tracelimit::info_ratelimited!(?crash, "Guest has reported system crash");

                self.partition.crash_notification_send.send(crash);
            }
            hvdef::HV_X64_MSR_GUEST_CRASH_P0
            | hvdef::HV_X64_MSR_GUEST_CRASH_P1
            | hvdef::HV_X64_MSR_GUEST_CRASH_P2
            | hvdef::HV_X64_MSR_GUEST_CRASH_P3
            | hvdef::HV_X64_MSR_GUEST_CRASH_P4 => {
                self.crash_reg[(msr - hvdef::HV_X64_MSR_GUEST_CRASH_P0) as usize] = value;
            }
            _ => return Err(MsrError::Unknown),
        }
        Ok(())
    }

    #[cfg(guest_arch = "x86_64")]
    fn read_msr(&mut self, msr: u32) -> Result<u64, MsrError> {
        let last_vtl = self.last_vtl();
        if msr & 0xf0000000 == 0x40000000 {
            if let Some(hv) = &mut self.hv(last_vtl) {
                let r = hv.msr_read(msr);
                if !matches!(r, Err(MsrError::Unknown)) {
                    return r;
                }
            }
        }

        let v = match msr {
            hvdef::HV_X64_MSR_GUEST_CRASH_CTL => self.crash_control.into(),
            hvdef::HV_X64_MSR_GUEST_CRASH_P0 => self.crash_reg[0],
            hvdef::HV_X64_MSR_GUEST_CRASH_P1 => self.crash_reg[1],
            hvdef::HV_X64_MSR_GUEST_CRASH_P2 => self.crash_reg[2],
            hvdef::HV_X64_MSR_GUEST_CRASH_P3 => self.crash_reg[3],
            hvdef::HV_X64_MSR_GUEST_CRASH_P4 => self.crash_reg[4],
            _ => return Err(MsrError::Unknown),
        };
        Ok(v)
    }

    /// Emulates an instruction due to a memory access exit.
    #[cfg(guest_arch = "x86_64")]
    async fn emulate<D: CpuIo>(
        &mut self,
        devices: &D,
        interruption_pending: bool,
    ) -> Result<(), VpHaltReason<UhRunVpError>>
    where
        for<'b> UhEmulationState<'b, 'a, D, T>:
            virt_support_x86emu::emulate::EmulatorSupport<Error = UhRunVpError>,
    {
        let guest_memory = self.last_vtl_gm();
        virt_support_x86emu::emulate::emulate(
            &mut UhEmulationState {
                vp: &mut *self,
                interruption_pending,
                devices,
            },
            guest_memory,
            devices,
        )
        .await
    }

    /// Emulates an instruction due to a memory access exit.
    #[cfg(guest_arch = "aarch64")]
    async fn emulate<D: CpuIo>(
        &mut self,
        devices: &D,
        intercept_state: &aarch64emu::InterceptState,
    ) -> Result<(), VpHaltReason<UhRunVpError>>
    where
        for<'b> UhEmulationState<'b, 'a, D, T>:
            virt_support_aarch64emu::emulate::EmulatorSupport<Error = UhRunVpError>,
    {
        let guest_memory = self.last_vtl_gm();
        virt_support_aarch64emu::emulate::emulate(
            &mut UhEmulationState {
                vp: &mut *self,
                interruption_pending: intercept_state.interruption_pending,
                devices,
            },
            intercept_state,
            guest_memory,
            devices,
        )
        .await
    }

    fn vtl1_supported(&self) -> bool {
        !matches!(
            *self.partition.guest_vsm.read(),
            GuestVsmState::NotPlatformSupported
        )
    }

    fn last_vtl(&self) -> Vtl {
        T::last_vtl(self)
    }

    /// Returns the guest memory object that should be used based on the last vtl
    fn last_vtl_gm(&self) -> &'a GuestMemory {
        &self.partition.gm[self.last_vtl()]
    }

    fn hv(&self, vtl: Vtl) -> Option<&ProcessorVtlHv> {
        self.hv[vtl].as_ref()
    }

    #[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
    fn hv_mut(&mut self, vtl: Vtl) -> Option<&mut ProcessorVtlHv> {
        self.hv[vtl].as_mut()
    }

    fn deliver_synic_messages(&mut self, vtl: Vtl, sints: u16) {
        let proxied_sints = self.hv(vtl).map_or(!0, |hv| hv.synic.proxied_sints());
        let pending_sints =
            self.inner.message_queues[vtl].post_pending_messages(sints, |sint, message| {
                if proxied_sints & (1 << sint) != 0 {
                    if let Some(synic) = &mut self.untrusted_synic {
                        synic.post_message(
                            &self.partition.gm[vtl],
                            sint,
                            message,
                            &mut self
                                .partition
                                .synic_interrupt(self.inner.vp_info.base.vp_index, vtl),
                        )
                    } else {
                        self.partition.hcl.post_message_direct(
                            self.inner.vp_info.base.vp_index.index(),
                            sint,
                            message,
                        )
                    }
                } else {
                    self.hv[vtl].as_mut().unwrap().synic.post_message(
                        &self.partition.gm[vtl],
                        sint,
                        message,
                        &mut self
                            .partition
                            .synic_interrupt(self.inner.vp_info.base.vp_index, vtl),
                    )
                }
            });

        self.request_sint_notifications(vtl, pending_sints);
    }

    #[cfg_attr(guest_arch = "aarch64", allow(dead_code))]
    fn switch_vtl(&mut self, target_vtl: Vtl) {
        T::switch_vtl_state(self, target_vtl);

        self.runner.set_exit_vtl(target_vtl);
    }
}

fn signal_mnf(dev: &impl CpuIo, connection_id: u32) {
    if let Err(err) = dev.signal_synic_event(Vtl::Vtl0, connection_id, 0) {
        tracelimit::warn_ratelimited!(
            error = &err as &dyn std::error::Error,
            connection_id,
            "failed to signal mnf"
        );
    }
}

/// Yields execution back to the executor.
async fn yield_now() {
    let mut yielded = false;
    poll_fn(|cx| {
        if !yielded {
            // Wake the waker so that this task gets to run again.
            cx.waker().wake_by_ref();
            yielded = true;
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await;
}

#[cfg(guest_arch = "x86_64")]
fn from_seg(reg: HvX64SegmentRegister) -> x86defs::SegmentRegister {
    x86defs::SegmentRegister {
        base: reg.base,
        limit: reg.limit,
        selector: reg.selector,
        attributes: reg.attributes.into(),
    }
}

struct UhEmulationState<'a, 'b, T: CpuIo, U: Backing> {
    vp: &'a mut UhProcessor<'b, U>,
    interruption_pending: bool,
    devices: &'a T,
}

struct UhHypercallHandler<'a, 'b, T, B: Backing> {
    vp: &'a mut UhProcessor<'b, B>,
    bus: &'a T,
    /// Indicates if the handler is for trusted hypercalls in case hardware isolation is in use. A
    /// hypercall is trusted if it was made by the guest using a regular vmcall instruction, without
    /// using any host-visible mechanisms. An untrusted hypercall was intercepted from the
    /// hypervisor, such as one made by the guest using an isolated mechanism such as tdcall or
    /// GHCB.
    ///
    /// This should always be false if hardware isolation is not in use, as the distinction does
    /// not exist in that case.
    trusted: bool,
}

impl<T, B: Backing> hv1_hypercall::GetVpIndexFromApicId for UhHypercallHandler<'_, '_, T, B> {
    fn get_vp_index_from_apic_id(
        &mut self,
        partition_id: u64,
        target_vtl: Vtl,
        apic_ids: &[u32],
        vp_indices: &mut [u32],
    ) -> hvdef::HvRepResult {
        tracing::debug!(partition_id, ?target_vtl, "HvGetVpIndexFromApicId");

        if partition_id != hvdef::HV_PARTITION_ID_SELF {
            return Err((HvError::InvalidPartitionId, 0));
        }

        if self.vp.last_vtl() < target_vtl {
            return Err((HvError::AccessDenied, 0));
        }

        #[cfg(guest_arch = "aarch64")]
        if true {
            let _ = apic_ids;
            let _ = vp_indices;
            todo!("AARCH64_TODO");
        }

        #[cfg(guest_arch = "x86_64")]
        for (i, (&apic_id, vp_index)) in apic_ids.iter().zip(vp_indices).enumerate() {
            *vp_index = self
                .vp
                .partition
                .vps
                .iter()
                .find(|vp| vp.vp_info.apic_id == apic_id)
                .ok_or((HvError::InvalidParameter, i))?
                .vp_info
                .base
                .vp_index
                .index()
        }

        Ok(())
    }
}

#[cfg(guest_arch = "aarch64")]
impl<T: CpuIo, B: Backing> Arm64RegisterState for UhHypercallHandler<'_, '_, T, B> {
    fn pc(&mut self) -> u64 {
        self.vp
            .runner
            .get_vp_register(HvArm64RegisterName::XPc)
            .expect("get vp register cannot fail")
            .as_u64()
    }

    fn set_pc(&mut self, pc: u64) {
        self.vp
            .runner
            .set_vp_register(HvArm64RegisterName::XPc, pc.into())
            .expect("set vp register cannot fail");
    }

    fn x(&mut self, n: u8) -> u64 {
        self.vp
            .runner
            .get_vp_register(HvArm64RegisterName(HvArm64RegisterName::X0.0 + n as u32))
            .expect("get vp register cannot fail")
            .as_u64()
    }

    fn set_x(&mut self, n: u8, v: u64) {
        self.vp
            .runner
            .set_vp_register(
                HvArm64RegisterName(HvArm64RegisterName::X0.0 + n as u32),
                v.into(),
            )
            .expect("set vp register cannot fail")
    }
}

impl<T, B: Backing> hv1_hypercall::StartVirtualProcessor<hvdef::hypercall::InitialVpContextX64>
    for UhHypercallHandler<'_, '_, T, B>
{
    fn start_virtual_processor(
        &mut self,
        partition_id: u64,
        target_vp: u32,
        target_vtl: Vtl,
        vp_context: &hvdef::hypercall::InitialVpContextX64,
    ) -> hvdef::HvResult<()> {
        tracing::debug!(
            vp_index = self.vp.vp_index().index(),
            target_vp,
            ?target_vtl,
            "HvStartVirtualProcessor"
        );

        if partition_id != hvdef::HV_PARTITION_ID_SELF {
            return Err(HvError::InvalidPartitionId);
        }

        if target_vp == self.vp.vp_index().index()
            || target_vp as usize >= self.vp.partition.vps.len()
        {
            return Err(HvError::InvalidVpIndex);
        }

        if self.vp.last_vtl() < target_vtl {
            return Err(HvError::AccessDenied);
        }

        let target_vp = &self.vp.partition.vps[target_vp as usize];

        // TODO CVM GUEST VSM: probably some validation on vtl1_enabled
        *target_vp.hv_start_enable_vtl_vp[target_vtl].lock() = Some(Box::new(*vp_context));
        target_vp.wake(target_vtl, WakeReason::HV_START_ENABLE_VP_VTL);

        Ok(())
    }
}

impl<T: CpuIo, B: Backing> hv1_hypercall::PostMessage for UhHypercallHandler<'_, '_, T, B> {
    fn post_message(&mut self, connection_id: u32, message: &[u8]) -> hvdef::HvResult<()> {
        tracing::trace!(
            connection_id,
            self.trusted,
            "handling post message intercept"
        );

        self.bus
            .post_synic_message(self.vp.last_vtl(), connection_id, self.trusted, message)
    }
}

impl<T: CpuIo, B: Backing> hv1_hypercall::SignalEvent for UhHypercallHandler<'_, '_, T, B> {
    fn signal_event(&mut self, connection_id: u32, flag: u16) -> hvdef::HvResult<()> {
        tracing::trace!(connection_id, "handling signal event intercept");

        self.bus
            .signal_synic_event(self.vp.last_vtl(), connection_id, flag)
    }
}

impl<T: CpuIo, B: Backing> UhHypercallHandler<'_, '_, T, B> {
    fn retarget_virtual_interrupt(
        &mut self,
        device_id: u64,
        address: u64,
        data: u32,
        vector: u32,
        multicast: bool,
        target_processors: &[u32],
    ) -> hvdef::HvResult<()> {
        let vpci_params = vmcore::vpci_msi::VpciInterruptParameters {
            vector,
            multicast,
            target_processors,
        };

        self.vp
            .partition
            .software_devices
            .as_ref()
            .expect("should exist if this intercept is registered or this is a CVM")
            .retarget_interrupt(device_id, address, data, &vpci_params)
    }
}

impl<T: CpuIo, B: Backing> hv1_hypercall::ModifySparseGpaPageHostVisibility
    for UhHypercallHandler<'_, '_, T, B>
{
    fn modify_gpa_visibility(
        &mut self,
        partition_id: u64,
        visibility: HostVisibilityType,
        gpa_pages: &[u64],
    ) -> hvdef::HvRepResult {
        if partition_id != hvdef::HV_PARTITION_ID_SELF {
            return Err((HvError::AccessDenied, 0));
        }

        tracing::debug!(
            ?visibility,
            pages = gpa_pages.len(),
            "modify_gpa_visibility"
        );

        let shared = match visibility {
            HostVisibilityType::PRIVATE => false,
            HostVisibilityType::SHARED => true,
            _ => return Err((HvError::InvalidParameter, 0)),
        };

        self.vp
            .partition
            .isolated_memory_protector
            .as_ref()
            .ok_or((HvError::AccessDenied, 0))?
            .change_host_visibility(shared, gpa_pages)
    }
}

impl<T: CpuIo, B: Backing> hv1_hypercall::QuerySparseGpaPageHostVisibility
    for UhHypercallHandler<'_, '_, T, B>
{
    fn query_gpa_visibility(
        &mut self,
        partition_id: u64,
        gpa_pages: &[u64],
        host_visibility: &mut [HostVisibilityType],
    ) -> hvdef::HvRepResult {
        if partition_id != hvdef::HV_PARTITION_ID_SELF {
            return Err((HvError::AccessDenied, 0));
        }

        self.vp
            .partition
            .isolated_memory_protector
            .as_ref()
            .ok_or((HvError::AccessDenied, 0))?
            .query_host_visibility(gpa_pages, host_visibility)
    }
}

impl<T, B: Backing> hv1_hypercall::ExtendedQueryCapabilities for UhHypercallHandler<'_, '_, T, B> {
    fn query_extended_capabilities(&mut self) -> hvdef::HvResult<u64> {
        // This capability is not actually supported. However Windows may unconditionally issue this
        // hypercall. Return InvalidHypercallCode as the error status. This is the same as not
        // implementing this at all, but has the advantage of not causing generating error messages.
        Err(HvError::InvalidHypercallCode)
    }
}
