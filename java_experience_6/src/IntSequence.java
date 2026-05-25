import java.util.Random;

// IntSequence 接口
interface IntSequence {
    boolean hasNext();
    int next();
}

// RandomIntSequence 类实现 IntSequence 接口
class RandomIntSequence implements IntSequence {
    private int n;
    private Random random = new Random();

    @Override
    public boolean hasNext() {
        n = random.nextInt(90) + 10;  // 生成10-99的随机两位数
        return true;
    }

    @Override
    public int next() {
        return n;
    }
}

// SequenceTest 类
class SequenceTest {
    public double average(RandomIntSequence rs, int m) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            rs.hasNext();  // 生成新的随机数
            sum += rs.next();  // 获取该随机数
        }
        return (double) sum / m;
    }
}