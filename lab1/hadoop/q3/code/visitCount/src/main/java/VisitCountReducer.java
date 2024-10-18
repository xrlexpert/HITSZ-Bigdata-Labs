import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class VisitCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable totalVisits = new IntWritable();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable value : values) {
            sum += value.get(); // 累加访问次数
        }
        totalVisits.set(sum);
        context.write(key, totalVisits); // 输出日期及其访问总次数
    }
}

