import grequests
import time

SOURCE = "public int[] twoSum(int[] nums, int target) {" \
         "PRED " \
         "for (int i = 0; i < n; ++i) {" \
         "for (int j = i + 1; j < n; ++j) {" \
         "if (nums[i] + nums[j] == target) {" \
         "return new int[]{i, j};}}}" \
         "return new int[0];}"

SOURCE2 = "public void map(Text key, LongWritable value, OutputCollector<Text, Text> output,Reporter reporter) throws IOException {" \
          "String name = key.toString();" \
          "long longValue = value.get();" \
          """reporter.setStatus("starting " + name + " ::host = " + hostName);""" \
          "PRED " \
          "parseLogFile(fs, new Path(name), longValue, output, reporter);" \
          "long tEnd = System.currentTimeMillis();" \
          "long execTime = tEnd - tStart;" \
          """reporter.setStatus("finished " + name + " ::host = " + hostName + " in " + execTime / 1000 + " sec.");}"""

url = "http://127.0.0.1:5000/"


# req_list = [ # 请求列表（慎重运行）
#     grequests.post(url, data=SOURCE),
#     grequests.post(url, data=SOURCE2),
#     grequests.post(url, data=SOURCE),
#     grequests.post(url, data=SOURCE2),
# ]

# 并发请求的数量
REQ_NUM = 10

start_time = time.time()
req_list = [grequests.post(url, data=SOURCE) for _ in range(REQ_NUM)]
res_list = grequests.map(req_list)    # 并行发送，等最后一个运行完后返回
print("total time: ", time.time()-start_time)
print(res_list[0].text)  # 打印第一个请求的响应文本
print(res_list[1].text)
print(res_list[2].text)