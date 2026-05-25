public class main {
    public static void main(String[] args) {
        RandomIntSequence sequence = new RandomIntSequence();
        SequenceTest tester = new SequenceTest();

        // 测试前5个数的平均值
        double avg1 = tester.average(sequence, 5);
        System.out.println("前5个随机数的平均值: " + avg1);

        // 测试前10个数的平均值
        double avg2 = tester.average(sequence, 10);
        System.out.println("前10个随机数的平均值: " + avg2);

        // 或者直接使用RandomIntSequence
        RandomIntSequence seq = new RandomIntSequence();
        System.out.println("\n生成3个随机数:");
        for (int i = 0; i < 3; i++) {
            seq.hasNext();
            System.out.println("第" + (i+1) + "个随机数: " + seq.next());
        }
    }
}