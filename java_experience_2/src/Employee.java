// 定义Employee类
class Employee {
    // 属性
    private String name;
    private int age;
    private double salary;
    private String employeeId;

    // 构造函数
    public Employee() {
    }

    public Employee(String name, int age, double salary, String employeeId) {
        this.name = name;
        this.age = age;
        this.salary = salary;
        this.employeeId = employeeId;
    }

    // 设置器（Setter）方法
    public void setName(String name) {
        this.name = name;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public void setSalary(double salary) {
        this.salary = salary;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
    }

    // 获取器（Getter）方法
    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public double getSalary() {
        return salary;
    }

    public String getEmployeeId() {
        return employeeId;
    }

    // sayHello方法
    public void sayHello() {
        System.out.println("My name is " + name);
    }
}
