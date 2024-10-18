import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class MaxScoreReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable maxScore = new IntWritable();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int max = Integer.MIN_VALUE; // 初始化最大值
        for (IntWritable value : values) {
            max = Math.max(max, value.get()); // 找到最高分
        }
        maxScore.set(max);
        context.write(key, maxScore); // 输出每个课程的最高分
    }
}
