// Copyright (C) Microsoft Corporation. All rights reserved.

//! Backing for SNP partitions.

use super::hcl_pvalidate_pages;
use super::hcl_rmpadjust_pages;
use super::mshv_pvalidate;
use super::mshv_rmpadjust;
use super::HclVp;
use super::MshvVtl;
use super::NoRunner;
use super::ProcessorRunner;
use crate::vmsa::VmsaWrapper;
use hvdef::HvRegisterName;
use hvdef::HvRegisterValue;
use hvdef::Vtl;
use hvdef::HV_PAGE_SIZE;
use memory_range::MemoryRange;
use sidecar_client::SidecarVp;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;
use thiserror::Error;
use vtl_array::VtlArray;
use x86defs::snp::SevRmpAdjust;
use x86defs::snp::SevVmsa;

/// Runner backing for SNP partitions.
pub struct Snp {
    vmsa: VtlArray<NonNull<SevVmsa>, 2>,
}

/// Error returned by failing SNP operations.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum SnpError {
    #[error("operating system error")]
    Os(#[source] nix::Error),
    #[error("isa error {0:?}")]
    Isa(u32),
}

/// Error returned by failing SNP page operations.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum SnpPageError {
    #[error("pvalidate failed")]
    Pvalidate(SnpError),
    #[error("rmpadjust failed")]
    Rmpadjust(SnpError),
}

impl MshvVtl {
    /// Execute the pvalidate instruction on the specified memory range.
    ///
    /// The range must not be mapped in the kernel as RAM.
    //
    // TODO SNP: figure out a safer model for this here and in the kernel.
    pub fn pvalidate_pages(
        &self,
        range: MemoryRange,
        validate: bool,
        terminate_on_failure: bool,
    ) -> Result<(), SnpPageError> {
        tracing::debug!(%range, validate, terminate_on_failure, "pvalidate");
        // SAFETY: TODO SNP: we are passing parameters as the kernel requires.
        // But this isn't really safe because it could be used to unaccept a
        // VTL2 kernel page. Kernel changes are needed to make this safe.
        let ret = unsafe {
            hcl_pvalidate_pages(
                self.file.as_raw_fd(),
                &mshv_pvalidate {
                    start_pfn: range.start() / HV_PAGE_SIZE,
                    page_count: (range.end() - range.start()) / HV_PAGE_SIZE,
                    validate: validate as u8,
                    terminate_on_failure: terminate_on_failure as u8,
                    ram: 0,
                    padding: [0; 1],
                },
            )
            .map_err(SnpError::Os)
            .map_err(SnpPageError::Pvalidate)?
        };

        if ret != 0 {
            return Err(SnpPageError::Pvalidate(SnpError::Isa(ret as u32)));
        }

        Ok(())
    }

    /// Execute the rmpadjust instruction on the specified memory range.
    ///
    /// The range must not be mapped in the kernel as RAM.
    //
    // TODO SNP: figure out a safer model for this here and in the kernel.
    pub fn rmpadjust_pages(
        &self,
        range: MemoryRange,
        value: SevRmpAdjust,
        terminate_on_failure: bool,
    ) -> Result<(), SnpPageError> {
        if value.vmsa() {
            // TODO SNP: VMSA conversion does not work.
            return Ok(());
        }

        #[allow(clippy::undocumented_unsafe_blocks)] // TODO SNP
        let ret = unsafe {
            hcl_rmpadjust_pages(
                self.file.as_raw_fd(),
                &mshv_rmpadjust {
                    start_pfn: range.start() / HV_PAGE_SIZE,
                    page_count: (range.end() - range.start()) / HV_PAGE_SIZE,
                    value: value.into(),
                    terminate_on_failure: terminate_on_failure as u8,
                    ram: 0,
                    padding: Default::default(),
                },
            )
            .map_err(SnpError::Os)
            .map_err(SnpPageError::Rmpadjust)?
        };

        if ret != 0 {
            return Err(SnpPageError::Rmpadjust(SnpError::Isa(ret as u32)));
        }

        Ok(())
    }
}

impl super::private::BackingPrivate for Snp {
    fn new(vp: &HclVp, sidecar: Option<&SidecarVp<'_>>) -> Result<Self, NoRunner> {
        assert!(sidecar.is_none());
        let super::BackingState::Snp { vmsa, vmsa_vtl1 } = &vp.backing else {
            return Err(NoRunner::MismatchedIsolation);
        };

        Ok(Self {
            vmsa: VtlArray::from([vmsa.0, vmsa_vtl1.0]),
        })
    }

    fn try_set_reg(
        _runner: &mut ProcessorRunner<'_, Self>,
        _name: HvRegisterName,
        _value: HvRegisterValue,
    ) -> Result<bool, super::Error> {
        Ok(false)
    }

    fn must_flush_regs_on(_runner: &ProcessorRunner<'_, Self>, _name: HvRegisterName) -> bool {
        false
    }

    fn try_get_reg(
        _runner: &ProcessorRunner<'_, Self>,
        _name: HvRegisterName,
    ) -> Result<Option<HvRegisterValue>, super::Error> {
        Ok(None)
    }
}

impl ProcessorRunner<'_, Snp> {
    /// Gets a reference to the VMSA and backing state of a VTL
    pub fn vmsa(&self, vtl: Vtl) -> VmsaWrapper<'_, &SevVmsa> {
        // SAFETY: the VMSA will not be concurrently accessed by the processor
        // while this VP is in VTL2.
        let vmsa = unsafe { &*(self.state.vmsa[vtl]).as_ptr() };

        VmsaWrapper::new(vmsa, &self.hcl.snp_register_bitmap)
    }

    /// Gets a mutable reference to the VMSA and backing state of a VTL.
    pub fn vmsa_mut(&mut self, vtl: Vtl) -> VmsaWrapper<'_, &mut SevVmsa> {
        // SAFETY: the VMSA will not be concurrently accessed by the processor
        // while this VP is in VTL2.
        let vmsa = unsafe { &mut *(self.state.vmsa[vtl]).as_ptr() };

        VmsaWrapper::new(vmsa, &self.hcl.snp_register_bitmap)
    }

    /// Gets references to multiple VMSAs for copying state
    pub fn vmsas_for_copy(
        &mut self,
        source_vtl: Vtl,
        target_vtl: Vtl,
    ) -> (VmsaWrapper<'_, &SevVmsa>, VmsaWrapper<'_, &mut SevVmsa>) {
        // SAFETY: the VMSA will not be concurrently accessed by the processor
        // while this VP is in VTL2.
        let (source_vmsa, target_vmsa) = unsafe {
            (
                &*(self.state.vmsa[source_vtl]).as_ptr(),
                &mut *(self.state.vmsa[target_vtl]).as_ptr(),
            )
        };

        (
            VmsaWrapper::new(source_vmsa, &self.hcl.snp_register_bitmap),
            VmsaWrapper::new(target_vmsa, &self.hcl.snp_register_bitmap),
        )
    }
}
