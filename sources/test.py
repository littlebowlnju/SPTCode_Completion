import requests


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

data = SOURCE
response = requests.post("{}/".format("http://127.0.0.1:8000"), json =data )
print(str(response.json()))