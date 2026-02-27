import ctypes
import os
import platform
import subprocess


def _ram_info_psutil():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "ok": True,
            "available_bytes": int(vm.available),
            "total_bytes": int(vm.total),
            "source": "psutil",
        }
    except Exception as e:
        return {"ok": False, "reason": f"psutil failed: {e}"}


def _ram_info_windows_ctypes():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return {
            "ok": True,
            "available_bytes": int(stat.ullAvailPhys),
            "total_bytes": int(stat.ullTotalPhys),
            "source": "windows_ctypes",
        }
    return {"ok": False, "reason": "GlobalMemoryStatusEx failed"}


def _ram_info_linux_proc():
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return {"ok": False, "reason": "/proc/meminfo not found"}
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            num = v.strip().split()[0]
            if num.isdigit():
                values[k] = int(num) * 1024
    if "MemAvailable" in values and "MemTotal" in values:
        return {
            "ok": True,
            "available_bytes": int(values["MemAvailable"]),
            "total_bytes": int(values["MemTotal"]),
            "source": "linux_proc_meminfo",
        }
    return {"ok": False, "reason": "MemAvailable/MemTotal not found"}


def _ram_info_macos_vmstat():
    try:
        total = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        vm_out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        for line in vm_out.splitlines():
            if "page size of" in line:
                parts = line.split("page size of", 1)[1].strip().split(" ", 1)
                if parts and parts[0].isdigit():
                    page_size = int(parts[0])
                break

        free_pages = 0
        inactive_pages = 0
        speculative_pages = 0
        for line in vm_out.splitlines():
            norm = line.strip().replace(".", "")
            if ":" not in norm:
                continue
            key, val = norm.split(":", 1)
            num = val.strip().split(" ")[0].replace(".", "")
            if not num.isdigit():
                continue
            n = int(num)
            if key == "Pages free":
                free_pages = n
            elif key == "Pages inactive":
                inactive_pages = n
            elif key == "Pages speculative":
                speculative_pages = n

        available = (free_pages + inactive_pages + speculative_pages) * page_size
        return {
            "ok": True,
            "available_bytes": int(available),
            "total_bytes": int(total),
            "source": "macos_vm_stat",
        }
    except Exception as e:
        return {"ok": False, "reason": f"vm_stat failed: {e}"}


def get_ram_info():
    probe = _ram_info_psutil()
    if probe.get("ok"):
        return probe

    system = platform.system()
    if system == "Windows":
        return _ram_info_windows_ctypes()
    if system == "Linux":
        return _ram_info_linux_proc()
    if system == "Darwin":
        return _ram_info_macos_vmstat()
    return {"ok": False, "reason": f"unsupported platform: {system}"}
