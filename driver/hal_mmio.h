/*============================================================================
 * hal_mmio.h - the HAL you would use on a real SoC (memory-mapped AXI-Lite).
 *
 * The accelerator's register bus (reg_wr_en / reg_rd_en / reg_addr / reg_wdata /
 * reg_rdata) is what an AXI4-Lite slave bridge produces. To the CPU the
 * accelerator then looks like an array of 32-bit registers at some base
 * address, and "talking to the accelerator" is just loads and stores:
 *
 *      *(volatile uint32_t *)(BASE + 4*word_addr) = value;      // write
 *      value = *(volatile uint32_t *)(BASE + 4*word_addr);      // read
 *
 * `volatile` tells the compiler these accesses have side effects and must not be
 * removed, merged or reordered. Word address -> byte address is x4 because AXI
 * addresses bytes while our register bus addresses 32-bit words.
 *
 * Usage (bare metal):
 *      dnn_mmio_t m = { (volatile uint32_t *)0x40000000u };
 *      dnn_hal_t hal = { &m, dnn_mmio_read, dnn_mmio_write, NULL };
 *      dnn_probe(&dev, &hal);
 *
 * Linux user space: mmap() /dev/mem (or a UIO device) and use the returned
 * pointer as `base`. A DMA engine is NOT needed: the weights are loaded once and
 * activations are a few hundred bytes per inference in this design.
 *
 * If your bridge inserts wait states or posts writes, add a memory barrier
 * (e.g. __sync_synchronize()) before the START write and after reading STATUS.
 *==========================================================================*/
#ifndef HAL_MMIO_H
#define HAL_MMIO_H
#include <stdint.h>

typedef struct { volatile uint32_t *base; } dnn_mmio_t;

static inline uint32_t dnn_mmio_read(void *ctx, uint16_t word_addr)
{
    return ((dnn_mmio_t *)ctx)->base[word_addr];        /* base is uint32_t*: index = word */
}

static inline void dnn_mmio_write(void *ctx, uint16_t word_addr, uint32_t data)
{
    ((dnn_mmio_t *)ctx)->base[word_addr] = data;
}

#endif /* HAL_MMIO_H */
