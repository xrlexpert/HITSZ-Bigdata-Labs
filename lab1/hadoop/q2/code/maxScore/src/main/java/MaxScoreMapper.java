import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class MaxScoreMapper extends Mapper<Object, Text, Text, IntWritable> {
    private Text subject = new Text();
    private IntWritable score = new IntWritable();
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\\s+");
        if (parts.length == 2) {
            subject.set(parts[0]); // 课程
            score.set(Integer.parseInt(parts[1])); // 成绩
            context.write(subject, score);
        }
    }
}
