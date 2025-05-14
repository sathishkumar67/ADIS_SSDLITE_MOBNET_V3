# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# plt.imshow(images[2].cpu().permute(1, 2, 0).int().numpy())
# # plot the bounding boxes
# x1, y1, x2, y2 = box
# width = x2 - x1
# height = y2 - y1
# rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
# plt.gca().add_patch(rect)
# # axis off
# plt.axis('off')
# plt.show()