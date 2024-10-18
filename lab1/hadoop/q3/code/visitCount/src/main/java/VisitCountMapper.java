import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class VisitCountMapper extends Mapper<Object, Text, Text, IntWritable> {
    private Text date = new Text();
    private IntWritable one = new IntWritable(1);
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        // 假设每行格式为 "field1,field2,field3,field4,date,field6"
        String[] fields = value.toString().split(",");
        if (fields.length > 5) {
            String visitTime = fields[4];
            String dateTime = visitTime.split(" ")[0];
            date.set(dateTime); // 获取访问日期
            
            context.write(date, one); // 输出日期和计数1
        }
    }
}
