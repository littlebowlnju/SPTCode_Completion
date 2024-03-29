import grequests
from gevent import monkey as curious_george

# ------ original source ------------
# SOURCE = "public int[] twoSum(int[] nums, int target) {" \
#          "int n = nums.length;" \
#          "for (int i = 0; i < n; ++i) {" \
#          "for (int j = i + 1; j < n; ++j) {" \
#          "if (nums[i] + nums[j] == target) {" \
#          "return new int[]{i, j};}}}" \
#          "return new int[0];}"

# ------- source with PRED token -------------
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

req_list = [ # 请求列表（慎重运行）
    grequests.post("{}/".format("http://127.0.0.1:8000"), data=SOURCE),
    grequests.post("{}/".format("http://127.0.0.1:8000"), data=SOURCE2),
    grequests.post("{}/".format("http://127.0.0.1:8000"), data=SOURCE),
    grequests.post("{}/".format("http://127.0.0.1:8000"), data=SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE2),
    #grequests.post("{}/".format("http://127.0.0.1:8000"), SOURCE),
]

curious_george.patch_all(thread=False, select=False)
#urls = ["http://127.0.0.1:8000","http://127.0.0.1:8000","http://127.0.0.1:8000","http://127.0.0.1:8000"]
#dataset = [SOURCE,SOURCE2,SOURCE,SOURCE2]
#rs = (grequests.post(u, data=SOURCE2) for u in urls)

res_list = grequests.map(req_list)    # 并行发送，等最后一个运行完后返回
print(res_list[0].text)  # 打印第一个请求的响应文本
print(res_list[1].text)
print(res_list[2].text)
print(res_list[3].text)