/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * NaN and infinity in B10G11R11 images, compressed or not.
 *
 * Vulkan converts a NaN to a NaN of the unsigned 10- and 11-bit float formats,
 * and a bit-exact copy keeps it. Every write path stores a pattern with NaN
 * and +inf texels in each channel (vulkan_b10g11r11_nan_pattern.glsl), and
 * every read path checks each texel: NaN reads back as a NaN, +inf as +inf,
 * every other value exactly.
 *
 * Images: "storage" (STORAGE | SAMPLED | TRANSFER | COLOR_ATTACHMENT, ALIAS,
 * the shape of vkd3d-proton's placed UAV/render targets) and "colour"
 * (SAMPLED | TRANSFER | COLOR_ATTACHMENT).
 * Writes: render (clear, full-screen draw, store), render2 (render, then a
 * load/store pass drawing one texel), store (compute imageStore, storage
 * image only) and upload (vkCmdCopyBufferToImage of the packed words).
 * Reads: copy (vkCmdCopyImageToBuffer), fetch (texelFetch) and load
 * (imageLoad, storage image only).
 *
 * Run it with and without the driver compressing the images
 * (ASAHI_MESA_DEBUG=nocompress); AGX_MESA_DEBUG=perf shows which are.
 *
 * Exit status 0 = every texel of every case as expected, 1 = a mismatch,
 * 2 = setup failure.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_b10g11r11_nan_shaders.h"

#define CHECK(x)                                                                         \
   do {                                                                                  \
      VkResult r_ = (x);                                                                 \
      if (r_ != VK_SUCCESS) {                                                            \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);            \
         exit(2);                                                                        \
      }                                                                                  \
   } while (0)

#define W 128
#define H 96

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkPipeline p_store, p_read, p_read_tex, p_gfx;
static VkDescriptorPool dpool;
static VkSampler smp;

struct pc { uint32_t w, h, mode, nanbits; };

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

static VkCommandBuffer
begin(void)
{
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));
   CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
   return cb;
}

static void
submit(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb}, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void
barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

static void
to_general(VkCommandBuffer cb, VkImage img)
{
   VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib});
}

/* expected channel encodings: 0 normal (value), 1 NaN, 2 inf */
struct texel { unsigned kind[3]; uint32_t enc[3]; float f[3]; };

static struct texel
expect(uint32_t x, uint32_t y)
{
   struct texel t;
   uint32_t k = x & 15, j = y & 15, a = x & 7, b = y & 7;
   t.enc[0] = (13u << 6) | (k << 2);
   t.f[0] = 0.25f + k / 64.0f;
   t.enc[1] = 14u << 6;
   t.f[1] = 0.5f;
   t.enc[2] = (14u << 5) | (16u + j);
   t.f[2] = 0.75f + j / 64.0f;
   for (int c = 0; c < 3; ++c) t.kind[c] = 0;
   if (a == 3 && b == 5) t.kind[0] = 1;
   if (a == 1 && b == 1) t.kind[0] = 2;
   if (a == 5 && b == 3) t.kind[1] = 1;
   if (a == 7 && b == 0) t.kind[1] = 2;
   if (a == 6 && b == 6) t.kind[2] = 1;
   if (a == 2 && b == 7) t.kind[2] = 2;
   return t;
}

static uint32_t
pack_expected(uint32_t x, uint32_t y)
{
   struct texel t = expect(x, y);
   uint32_t e[3];
   for (int c = 0; c < 3; ++c) {
      unsigned mb = c == 2 ? 5 : 6;
      e[c] = t.kind[c] == 1 ? ((31u << mb) | (1u << (mb - 1))) : t.kind[c] == 2 ? (31u << mb) : t.enc[c];
   }
   return e[0] | (e[1] << 11) | (e[2] << 22);
}

struct tally { unsigned nan_ok, nan_n, inf_ok, inf_n, oth_ok, oth_n; char first[160]; };

static void
check_word(struct tally *T, uint32_t x, uint32_t y, uint32_t v)
{
   struct texel t = expect(x, y);
   uint32_t ch[3] = {v & 0x7ff, (v >> 11) & 0x7ff, v >> 22};
   for (int c = 0; c < 3; ++c) {
      unsigned mb = c == 2 ? 5 : 6;
      uint32_t ex = ch[c] >> mb, ma = ch[c] & ((1u << mb) - 1);
      bool ok;
      if (t.kind[c] == 1) { T->nan_n++; ok = ex == 31 && ma; T->nan_ok += ok; }
      else if (t.kind[c] == 2) { T->inf_n++; ok = ex == 31 && !ma; T->inf_ok += ok; }
      else { T->oth_n++; ok = ch[c] == t.enc[c]; T->oth_ok += ok; }
      if (!ok && !T->first[0])
         snprintf(T->first, sizeof(T->first), "%u,%u c%d word=0x%08x want=0x%08x", x, y, c, v, pack_expected(x, y));
   }
}

static void
check_float(struct tally *T, uint32_t x, uint32_t y, const uint32_t *v)
{
   struct texel t = expect(x, y);
   for (int c = 0; c < 3; ++c) {
      float f;
      memcpy(&f, &v[c], 4);
      bool ok;
      if (t.kind[c] == 1) { T->nan_n++; ok = isnan(f); T->nan_ok += ok; }
      else if (t.kind[c] == 2) { T->inf_n++; ok = isinf(f) && f > 0; T->inf_ok += ok; }
      else { T->oth_n++; ok = f == t.f[c]; T->oth_ok += ok; }
      if (!ok && !T->first[0])
         snprintf(T->first, sizeof(T->first), "%u,%u c%d got=0x%08x/0x%08x/0x%08x", x, y, c, v[0], v[1], v[2]);
   }
}

/* returns 1 if any texel of the case is wrong */
static unsigned
print_tally(const char *kind, const char *wr, const char *rd, const struct tally *T)
{
   const bool ok = T->nan_ok == T->nan_n && T->inf_ok == T->inf_n && T->oth_ok == T->oth_n;
   printf("RESULT %-7s %-7s %-5s nan=%u/%u inf=%u/%u other=%u/%u %s%s%s\n", kind, wr, rd, T->nan_ok,
          T->nan_n, T->inf_ok, T->inf_n, T->oth_ok, T->oth_n, ok ? "PASS" : "FAIL",
          T->first[0] ? " first " : "", T->first);
   return !ok;
}

static VkShaderModule
sm(const uint32_t *code, size_t size)
{
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code}, NULL, &m));
   return m;
}

static VkPipeline
cpipe(const uint32_t *code, size_t size)
{
   VkPipeline p;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm(code, size), .pName = "main"},
      .layout = pl}, NULL, &p));
   return p;
}

static VkPipeline
gpipe(void)
{
   VkPipeline p;
   VkFormat f = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = sm(nan_vert, sizeof(nan_vert)), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = sm(nan_frag, sizeof(nan_frag)), .pName = "main"},
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                .colorAttachmentCount = 1, .pColorAttachmentFormats = &f},
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1.0f},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}},
      .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2,
         .pDynamicStates = dyn},
      .layout = pl}, NULL, &p));
   return p;
}

static const struct pc PC = {W, H, 0, 0x7fc00000u};

static void
draw(VkCommandBuffer cb, VkImageView v, bool load, VkRect2D sc)
{
   vkCmdBeginRendering(cb, &(VkRenderingInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {W, H}}, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_gfx);
   vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

/* returns the number of failing read paths */
static unsigned
run(int kind, int wpath)
{
   static const char *wname[] = {"render", "render2", "store", "upload"};
   const bool storage = kind == 0;
   if (wpath == 2 && !storage)
      return 0;
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = storage ? VK_IMAGE_CREATE_ALIAS_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0) | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img;
   CHECK(vkCreateImage(dev, &ci, NULL, &img));
   VkMemoryRequirements ir;
   vkGetImageMemoryRequirements(dev, img, &ir);
   VkBuffer alias;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = ir.size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &alias));
   VkMemoryRequirements ar;
   vkGetBufferMemoryRequirements(dev, alias, &ar);
   VkDeviceMemory im;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = ir.size > ar.size ? ir.size : ar.size,
      .memoryTypeIndex = mem_type(ir.memoryTypeBits & ar.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      NULL, &im));
   CHECK(vkBindImageMemory(dev, img, im, 0));
   CHECK(vkBindBufferMemory(dev, alias, im, 0));

   VkImageView v;
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ci.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &v));

   /* host buffer: [upload words | copy readback | float readback (16 B/texel) | raw dump] */
   const VkDeviceSize plane = (VkDeviceSize)W * H * 4, fplane = (VkDeviceSize)W * H * 16;
   const VkDeviceSize hsz = 2 * plane + fplane + ir.size;
   VkBuffer hb;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hsz,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, NULL, &hb));
   VkMemoryRequirements hr;
   vkGetBufferMemoryRequirements(dev, hb, &hr);
   VkDeviceMemory hm;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = hr.size, .memoryTypeIndex = mem_type(hr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &hm));
   CHECK(vkBindBufferMemory(dev, hb, hm, 0));
   uint8_t *hp;
   CHECK(vkMapMemory(dev, hm, 0, VK_WHOLE_SIZE, 0, (void **)&hp));
   memset(hp, 0, hsz);
   uint32_t *up = (uint32_t *)hp;
   for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
         up[y * W + x] = pack_expected(x, y);

   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
      .descriptorSetCount = 1, .pSetLayouts = &dsl}, &set));
   VkDescriptorImageInfo ii = {.sampler = smp, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   VkDescriptorBufferInfo bi = {hb, 2 * plane, fplane};
   VkWriteDescriptorSet wr[3] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi},
   };
   /* a storage descriptor of a non-storage image is invalid: the colour image binds only 1 and 2 */
   vkUpdateDescriptorSets(dev, storage ? 3 : 2, storage ? wr : wr + 1, 0, NULL);

   VkCommandBuffer cb = begin();
   to_general(cb, img);
   if (wpath <= 1) {
      draw(cb, v, false, (VkRect2D){{0, 0}, {W, H}});
      if (wpath == 1) {
         barrier(cb);
         draw(cb, v, true, (VkRect2D){{0, 0}, {1, 1}});
      }
   } else if (wpath == 2) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_store);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
      vkCmdDispatch(cb, W / 8, H / 8, 1);
   } else {
      vkCmdCopyBufferToImage(cb, hb, img, VK_IMAGE_LAYOUT_GENERAL, 1, &(VkBufferImageCopy){
         .bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}});
   }
   barrier(cb);
   vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &(VkBufferImageCopy){
      .bufferOffset = plane, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}});
   vkCmdCopyBuffer(cb, alias, hb, 1, &(VkBufferCopy){0, 2 * plane + fplane, ir.size});
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, storage ? p_read : p_read_tex);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
   vkCmdDispatch(cb, W / 8, H / 8, 1);
   submit(cb);

   const char *kn = storage ? "storage" : "colour";
   unsigned fails = 0;
   struct tally tc = {0}, tf = {0}, tl = {0};
   const uint32_t *cw = (const uint32_t *)(hp + plane), *fw = (const uint32_t *)(hp + 2 * plane);
   for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
         check_word(&tc, x, y, cw[y * W + x]);
         check_float(&tf, x, y, &fw[(y * W + x) * 4]);
      }
   fails += print_tally(kn, wname[wpath], "copy", &tc);
   fails += print_tally(kn, wname[wpath], "fetch", &tf);

   if (storage) {
      memset(hp + 2 * plane, 0, fplane);
      cb = begin();
      struct pc pc1 = PC;
      pc1.mode = 1;
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_read);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(pc1), &pc1);
      vkCmdDispatch(cb, W / 8, H / 8, 1);
      submit(cb);
      for (uint32_t y = 0; y < H; ++y)
         for (uint32_t x = 0; x < W; ++x)
            check_float(&tl, x, y, &fw[(y * W + x) * 4]);
      fails += print_tally(kn, wname[wpath], "load", &tl);
   }

   vkUnmapMemory(dev, hm);
   vkDestroyBuffer(dev, hb, NULL);
   vkFreeMemory(dev, hm, NULL);
   vkDestroyImageView(dev, v, NULL);
   vkDestroyBuffer(dev, alias, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, im, NULL);
   CHECK(vkResetDescriptorPool(dev, dpool, 0));
   return fails;
}

int
main(void)
{
   CHECK(vkCreateInstance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                               .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if (!n)
      return 2;
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
   uint32_t nq = 8;
   VkQueueFamilyProperties qf[8];
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qf);
   for (qfam = 0; qfam < nq && !(qf[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT); ++qfam)
      ;
   float prio = 1.0f;
   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
   CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1,
         .pQueuePriorities = &prio}}, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam}, NULL, &pool));
   VkDescriptorSetLayoutBinding b[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b}, NULL, &dsl));
   VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
                                 {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
   CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 3, .pPoolSizes = ps},
      NULL, &dpool));
   CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_ALL, 0, sizeof(struct pc)}}, NULL, &pl));
   CHECK(vkCreateSampler(dev, &(VkSamplerCreateInfo){.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST}, NULL, &smp));
   p_store = cpipe(store_comp, sizeof(store_comp));
   p_read = cpipe(load_comp, sizeof(load_comp));
   p_read_tex = cpipe(fetch_comp, sizeof(fetch_comp));
   p_gfx = gpipe();
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &fp);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   if ((fp.optimalTilingFeatures & need) != need) {
      printf("SKIP B10G11R11 lacks storage, sampled or colour attachment support\n");
      return 0;
   }
   printf("DEVICE %s\n", props.deviceName);
   unsigned fails = 0, cases = 0;
   for (int kind = 0; kind < 2; ++kind)
      for (int wpath = 0; wpath < 4; ++wpath) {
         if (wpath == 2 && kind)
            continue;
         fails += run(kind, wpath);
         cases += kind ? 2 : 3;
      }
   printf("SUMMARY %u/%u cases pass\n", cases - fails, cases);
   return fails ? 1 : 0;
}
