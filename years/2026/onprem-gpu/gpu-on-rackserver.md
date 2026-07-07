For most rack servers with a GPU, **Plan B is usually the best default**: install Linux on the host and run GPU workloads in Docker containers. It gives the least overhead, the simplest GPU access, and the cleanest path for ML, inference, transcoding, or batch processing. [serversimply](https://www.serversimply.com/blog/how-to-use-a-gpu-server)

## Which plan fits best

- **Plan A: Windows Server host + Linux VM** is the least attractive unless you specifically need Windows Server features or a Windows-only management stack. GPU access inside a VM usually needs extra virtualization features such as passthrough or partitioning, which adds complexity and can reduce flexibility. [docs.redhat](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/configuring_and_managing_virtualization/assembly_managing-gpu-devices-in-virtual-machines_configuring-and-managing-virtualization)
- **Plan B: Linux host + Docker** is the most practical for a single GPU server used for compute workloads. NVIDIA’s container approach is designed to let containers use the host GPU directly, so you avoid the cost and friction of running a full VM just to access the card. [forums.developer.nvidia](https://forums.developer.nvidia.com/t/nvidia-docker-gpu-server-application-deployment-made-easy/58076)
- **Plan C: Linux host + multiple Linux VMs** makes sense only when you need hard isolation between tenants, separate kernels, or different OS environments. For GPU sharing across multiple VMs, you generally need advanced passthrough or vGPU-style capabilities, and those are more limited and hardware-dependent. [forums.developer.nvidia](https://forums.developer.nvidia.com/t/how-to-get-two-vms-gpu-accelerated-on-my-physical-workstation/187841)

## My recommendation

If your goal is AI/ML, CUDA apps, media processing, or general GPU compute, choose **Linux host + Docker**. If you need multiple isolated environments for different teams or services, then consider **Linux host + multiple VMs**, but only if you already know your GPU and server platform support the needed virtualization mode. [virtualizationhowto](https://www.virtualizationhowto.com/2025/10/how-to-run-gpu-enabled-containers-in-your-home-lab/)

## Simple rule of thumb

- Single user, single GPU, compute workloads: **Plan B**.
- Need Windows-only apps or admin tools: **Plan A** only if necessary.
- Need strong isolation or many tenants: **Plan C**.

If you want, I can turn this into a concrete server blueprint for your exact GPU and workload, including Linux distro, Docker stack, and GPU driver setup.

***

To do **Plan C** (Linux host + multiple Linux VMs, each getting GPU access), you basically set up a **Linux KVM host (often Proxmox or plain libvirt/KVM)** and then either:

- **passthrough one physical GPU per VM**, or  
- use **NVIDIA‑styled vGPU** if your card and licenses support it, so one GPU can be sliced across multiple VMs.

Below is a practical, high‑level blueprint you can adapt for your rack server.

***

### 1. Pick the host stack

- Use **Ubuntu / Rocky Linux** with **KVM + libvirt** (or **Proxmox VE** if you want a web UI and cluster‑ready layout) as the host OS. [ubuntu](https://ubuntu.com/server/docs/how-to/graphics/gpu-virtualization-with-qemu-kvm/)
- Ensure your CPU and motherboard support **VT‑x + VT‑d / IOMMU** and that it’s enabled in BIOS. [github](https://github.com/clayfreeman/gpu-passthrough)

***

### 2. Enable IOMMU and VFIO

On the host, you need to:

1. Turn on IOMMU in the kernel via `GRUB_CMDLINE_LINUX`:
   - Intel: `intel_iommu=on`
   - AMD: `amd_iommu=on`
2. Load the `vfio‑pci` module and bind the GPU to it instead of the host driver. [davidyat](https://davidyat.es/2016/09/08/gpu-passthrough/)
3. Confirm the GPU is in a dedicated IOMMU group so it can be safely passed through to a VM. [github](https://github.com/clayfreeman/gpu-passthrough)

Tutorials like `gpu‑passthrough` (GitHub) give exact scripts and commands for this. [github](https://github.com/bryansteiner/gpu-passthrough-tutorial)

***

### 3. Create multiple Linux VMs

- Define each VM as a **KVM guest** (via `virt‑manager`, `virsh`, or Proxmox’s web UI). [dohost](https://dohost.us/index.php/2025/09/09/managing-gpu-passthrough-with-kvm-for-high-performance-applications/)
- Use **q35 machine type**, UEFI firmware if possible, and **virtio** for disk and network for best performance. [ubuntu](https://ubuntu.com/server/docs/how-to/graphics/gpu-virtualization-with-qemu-kvm/)

***

### 4. Assign GPUs to VMs

You have two main options:

#### Option A – One GPU per VM (passthrough)

- For each VM, add a **PCI host device** pointing to the GPU (and its HDMI audio if present). [youtube](https://www.youtube.com/watch?v=2aHQbg9j_gI)
- After the GPU is attached, install the appropriate **driver inside the guest** (e.g., NVIDIA proprietary driver). [forum.proxmox](https://forum.proxmox.com/threads/gpu-passthrough-with-nvidia-in-linux-vm-improve-stability.166766/)
- The GPU is then **exclusively owned by that VM**, giving near‑native performance. [reddit](https://www.reddit.com/r/VFIO/comments/1r4zrhl/1_gpu_for_multiple_vms_inside_linux/)

This is simplest if you have multiple GPUs or only a few VMs that need the GPU at a time.

#### Option B – Slice GPU across VMs (vGPU / MIG)

If you have **NVIDIA data‑center cards** (e.g., A100, A40, L40, etc.) and vGPU licenses:

- Enable **MIG (Multi‑Instance GPU)** or **NVIDIA vGPU** on the host. [docs.netapp](https://docs.netapp.com/us-en/flexpod/healthcare/flexpod-genomics-appendix-d-gpu-setup.html)
- Configure **partition profiles** (e.g., 1/2, 1/4 GPU) and attach a virtual GPU device to each VM. [nvidia](https://www.nvidia.com/en-us/data-center/virtual-solutions/)

This is more complex and license‑heavy, but lets you **over‑commit** GPU resources to many lightweight VMs.

***

### 5. Networking and storage for multiple VMs

- Use **bridged networking** or a **Linux bridge + VLANs** so each VM gets its own IPs and can be managed independently. [forum.proxmox](https://forum.proxmox.com/threads/gpu-passthrough-with-nvidia-in-linux-vm-improve-stability.166766/)
- Put VM disks on a **fast NVMe or SATA SSD**; if you want shared storage, add a **ZFS (for Proxmox)** or **LVM** pool. [blog.briancmoses](https://blog.briancmoses.com/2024/09/self-hosting-ai-with-spare-parts.html)

***

### 6. Manage headless GPU usage

For AI/ML, batch, or inference workloads:

- Install **NVIDIA drivers + CUDA** inside each GPU‑attached VM.  
- Use **systemd services or containers inside the VMs** to run workloads instead of a desktop session. [gist.github](https://gist.github.com/Markus92/6ef0c3157445ba09f5441adbbd9af12f)
- If you want job scheduling, you can wire a **slurm** or **batch system** across the VMs, treating each as a GPU node. [gist.github](https://gist.github.com/Markus92/6ef0c3157445ba09f5441adbbd9af12f)

***

### When to prefer Plan C vs Plan B

- Choose **Plan C** if you want **strong isolation**, per‑tenant VMs, or different OS distros, at the cost of more setup and management. [nvidia](https://www.nvidia.com/en-us/data-center/virtual-solutions/)
- Stick with **Plan B** if your main goal is **simple, low‑overhead GPU compute** (ML, FFmpeg, etc.) and you don’t need multiple full OS installs.

***

If you tell me your **exact GPU model (consumer vs data‑center)** and whether you prefer **Proxmox** or **plain KVM/libvirt**, I can give you a concrete step‑by‑step config (commands, XML snippets, driver choices).

***

NVIDIA’s vGPU licensing is **per‑user (CCU = Concurrent Connected User)** and can be **quite expensive**, especially for data‑center / workstation‑class vGPU tiers. [nvidia](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/Virtual-GPU-Packaging-and-Licensing-Guide.pdf)

***

### Typical list‑price ranges (USD)

These are approximate **per‑CCU / per year** (standard enterprise pricing; education discounts are lower): [docs.nvidia](https://docs.nvidia.com/vgpu/packaging-pricing-licensing-guide/latest/)

| vGPU product / tier              | Model type        | Approx. annual cost (per CCU) |
|----------------------------------|-------------------|-------------------------------|
| **NVIDIA Virtual Applications**  | Light GPU apps    | ~$10–20 per CCU  [docs.nvidia](https://docs.nvidia.com/vgpu/packaging-pricing-licensing-guide/latest/) |
| **NVIDIA Virtual PC**            | Basic desktop GPU | ~$50 per CCU  [docs.nvidia](https://docs.nvidia.com/vgpu/packaging-pricing-licensing-guide/latest/) |
| **NVIDIA RTX / vWS (workstation)** | High‑end 3D / CAD | ~$250+ per CCU  [nvidia](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/Virtual-GPU-Packaging-and-Licensing-Guide.pdf) |

- **Perpetual + SUMS** (Support + Maintenance): permanent license but with yearly renewal fees (~$5–$100 per CCU per year depending on tier). [images.nvidia](https://images.nvidia.com/content/vGPU/pdf/Virtual-GPU-Packaging-and-Licensing-Guide.pdf)
- **Annual subscription**: license + support bundled; often cheaper upfront than perpetual for short‑term use. [nvidia](https://www.nvidia.com/en-us/data-center/buy-grid/)

***

### Example total‑cost feel

- For **10 users** on **RTX Virtual Workstation**:
  - Perpetual model: around **$450 per CCU + ~$100 SUMS per CCU per year** → roughly **$4.5–5.5k total for 10 users** plus multi‑year SUMS. [ramprasadtech](https://ramprasadtech.com/wp-content/uploads/2023/03/NVIDIA-vGPU-Licensing-%E2%80%93-Deyda.net_.pdf)
  - Annual subscription: around **$250 per CCU per year** → **$2.5k/year for 10 users**. [docs.nvidia](https://docs.nvidia.com/vgpu/packaging-pricing-licensing-guide/latest/)

In practice, partners often bundle vGPU licenses with GPU hardware or cloud instances, and you can negotiate custom pricing or multi‑GPU bundles. [colfax-intl](https://colfax-intl.com/nvidia/nvidia-virtual-gpu-software)

***

### Rough ballpark for your use

If you’re slicing **one data‑center GPU (e.g., T4, A10, A100, L40)** across a few Linux VMs:

- Expect **$100–$500+ per CCU per year** depending on profile (compute vs high‑end 3D). [nvidia](https://www.nvidia.com/en-us/data-center/buy-grid/)
- For **5–10 VMs** (each treated as a CCU), that’s **$500–$5k+ per year** before any hardware cost. [ramprasadtech](https://ramprasadtech.com/wp-content/uploads/2023/03/NVIDIA-vGPU-Licensing-%E2%80%93-Deyda.net_.pdf)

If you tell me your **exact GPU model (e.g., A10, L40, A40, etc.)** and use case (AI training vs light inference vs desktop), I can narrow this to a concrete “per‑GPU‑per‑year” estimate.