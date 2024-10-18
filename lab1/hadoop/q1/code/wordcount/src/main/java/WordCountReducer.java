import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    int sum;
    IntWritable v = new IntWritable();
    HashMap<String,Integer> wordCountMap = new HashMap<>();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values,Context context) throws
            IOException, InterruptedException {
// 1 累加求和
        sum = 0;
        for (IntWritable count : values) {
            sum += count.get();
        }
// 2 输出
        v.set(sum);
        wordCountMap.put(key.toString(), sum);
    }
    @Override
    protected void cleanup(Context context) throws java.io.IOException, java.lang.InterruptedException
    {
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(
                (a, b) -> a.getValue().equals(b.getValue()) ? a.getKey().compareTo(b.getKey()) : a.getValue() - b.getValue()
        );
        for (Map.Entry<String, Integer> entry : wordCountMap.entrySet()) {
            pq.offer(entry);
            if (pq.size() > 20) {
                pq.poll();
            }
        }

        // 输出出现频率最高的 20 个关键词
        while (!pq.isEmpty()) {
            Map.Entry<String, Integer> entry = pq.poll();
            context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
        }
    }
}
