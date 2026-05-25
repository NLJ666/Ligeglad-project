class main {
    static void main(String[] args) {
        // 创建Employee对象（张三，28岁，工资5000，员工号A02）
        Employee employee = new Employee("张三", 28, 5000, "A02");

        // 调用sayHello方法
        employee.sayHello();

        // 测试其他方法
        System.out.println("员工号: " + employee.getEmployeeId());
        System.out.println("年龄: " + employee.getAge() + "岁");
        System.out.println("工资: " + employee.getSalary() + "元");

        // 测试设置器方法
//        employee.setName("李四");
//        employee.setAge(30);
//        employee.setSalary(6000);
//        employee.sayHello(); // 现在输出"My name is 李四"
    }
}
