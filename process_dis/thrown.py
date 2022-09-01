# def compute_equal(labels):
#     trans_labels = list(map(list, zip(*labels)))
#     lat = trans_labels[0]
#     lon = trans_labels[2]
#
#     count_lat = {}
#     for i in lat:
#         if count_lat.get(i) is None:
#             count_lat[i] = lat.count(i)
#
#     for key in list(count_lat.keys()):
#         if count_lat.get(key) < 10:
#             count_lat.pop(key)
#     print(count_lat)
#
#     count_lon = {}
#     for i in lon:
#         if count_lon.get(i) is None:
#             count_lon[i] = lon.count(i)
#
#     for key in list(count_lon.keys()):
#         if count_lon.get(key) < 10:
#             count_lon.pop(key)
#     print(count_lon)

# def draw(labels):
#     np_labels = np.array(labels)
#     np_labels = np_labels.transpose()
#     # 纬度
#     X = np_labels[0]
#     # 经度
#     Y = np_labels[2]
#
#     plt.figure(figsize=(16, 12))  # 定义图的大小
#     plt.title("Path")
#
#     plt.xlabel("x")
#     plt.ylabel("y")
#
#     plt.plot(X, Y)
#     plt.show()
