use bytemuck::cast_slice;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    U4,           // Packed 4-bit unsigned integers
    Svd,          // Low-rank SVD factorization
    ColumnarDict, // Column-major dictionary encoding
    Other,
}

pub enum TensorData {
    Mmap(Mmap),
    Memory(Vec<u8>),
}

impl TensorData {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TensorData::Mmap(m) => &m[..],
            TensorData::Memory(v) => &v[..],
        }
    }
}

#[derive(Clone)]
pub struct TensorPage {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    data: Arc<TensorData>,
    start: usize,                // Byte offset start
    end: usize,                  // Byte offset end
    pub svd_rank: Option<usize>, // SVD Rank if factorized
}

impl TensorPage {
    /// Loads a single file as a complete tensor page
    pub fn load_mmap_single(
        name: String,
        path: &Path,
        shape: Vec<usize>,
        dtype: DType,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe {
            let mut opts = memmap2::MmapOptions::new();
            // Try to use hugepages and populate the page tables immediately for latency reduction
            #[cfg(target_os = "linux")]
            opts.huge(None);

            // Note: populate is not always supported or might fail if hugepages are strict,
            // but we use standard map as a fallback if needed. For now, try populate.
            let _ = opts.populate();
            opts.map(&file)?
        };
        let end = mmap.len();

        Ok(TensorPage {
            name,
            shape,
            dtype,
            data: Arc::new(TensorData::Mmap(mmap)),
            start: 0,
            end,
            svd_rank: None,
        })
    }

    /// Creates a tensor page directly from a memory buffer.
    pub fn new_memory(name: String, shape: Vec<usize>, dtype: DType, buffer: Vec<u8>) -> Self {
        let end = buffer.len();
        TensorPage {
            name,
            shape,
            dtype,
            data: Arc::new(TensorData::Memory(buffer)),
            start: 0,
            end,
            svd_rank: None,
        }
    }

    /// Creates a tensor page as a slice of a larger memory-mapped file (e.g. safetensors)
    pub fn new_slice(
        name: String,
        data: Arc<TensorData>,
        shape: Vec<usize>,
        dtype: DType,
        start: usize,
        end: usize,
    ) -> Self {
        TensorPage {
            name,
            shape,
            dtype,
            data,
            start,
            end,
            svd_rank: None,
        }
    }

    /// Access the raw bytes of the tensor instantly.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data.as_bytes()[self.start..self.end]
    }

    /// Provide OS memory access pattern advice for this page.
    pub fn advise(&self, advice: memmap2::Advice) -> io::Result<()> {
        if let TensorData::Mmap(ref m) = *self.data {
            m.advise(advice)
        } else {
            Ok(())
        }
    }

    /// Provide OS memory access pattern advice for this page (Sequential + WillNeed) for BRM-S9-OPT-002.
    pub fn advise_prefetch(&self) -> io::Result<()> {
        if let TensorData::Mmap(ref m) = *self.data {
            let _ = m.advise(memmap2::Advice::Sequential);
            m.advise(memmap2::Advice::WillNeed)
        } else {
            Ok(())
        }
    }

    /// Advise the OS that this page's memory range is no longer needed (freed from RAM cache).
    pub fn dont_need(&self) -> io::Result<()> {
        if let TensorData::Mmap(ref m) = *self.data {
            unsafe { m.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) }
        } else {
            Ok(())
        }
    }

    /// Access as f32 slice (assumes tensor is actually f32)
    pub fn as_f32(&self) -> &[f32] {
        let b = self.as_bytes();
        assert!(
            (b.as_ptr() as usize) % 4 == 0,
            "CRITICAL ALIGNMENT ERROR: ptr is not 4-byte aligned!"
        );
        cast_slice(b)
    }

    /// Dynamic LoRA joining (adding an adapter to a base layer)
    pub fn join_f32(&self, lora: &TensorPage) -> Vec<f32> {
        let base = self.as_f32();
        let adapter = lora.as_f32();

        // Fast SIMD-friendly vector addition
        base.iter()
            .zip(adapter.iter())
            .map(|(b, a)| b + a)
            .collect()
    }
}
