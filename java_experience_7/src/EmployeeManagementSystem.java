// 1. 定义SalaryCalculator接口
interface SalaryCalculator {
    double calculateSalary();
    void displaySalaryDetails();
}

// 2. 定义抽象基类Employee
abstract class Employee {
    protected String employeeId;
    protected String name;
    protected String department;
    protected double baseSalary;

    // 构造方法
    public Employee(String employeeId, String name, String department, double baseSalary) {
        this.employeeId = employeeId;
        this.name = name;
        this.department = department;
        this.baseSalary = baseSalary;
    }

    // 抽象方法
    public abstract void work();

    // 具体方法
    public void displayInfo() {
        System.out.println("员工ID: " + employeeId);
        System.out.println("姓名: " + name);
        System.out.println("部门: " + department);
        System.out.println("基本工资: " + baseSalary + "元");
    }

    // Getter方法
    public String getEmployeeId() {
        return employeeId;
    }

    public String getName() {
        return name;
    }

    public String getDepartment() {
        return department;
    }

    public double getBaseSalary() {
        return baseSalary;
    }
}

// 3. 普通员工类 RegularEmployee
class RegularEmployee extends Employee implements SalaryCalculator {
    private int yearsOfService;
    private double performanceBonus;

    public RegularEmployee(String employeeId, String name, String department,
                           double baseSalary, int yearsOfService, double performanceBonus) {
        super(employeeId, name, department, baseSalary);
        this.yearsOfService = yearsOfService;
        this.performanceBonus = performanceBonus;
    }

    @Override
    public void work() {
        System.out.println("正在处理日常工作");
    }

    @Override
    public double calculateSalary() {
        // 总工资 = 基本工资 + 绩效奖金 + (工作年限 * 1000)
        return baseSalary + performanceBonus + (yearsOfService * 1000);
    }

    @Override
    public void displaySalaryDetails() {
        System.out.println("===== 工资明细 =====");
        System.out.println("基本工资: " + baseSalary + "元");
        System.out.println("绩效奖金: " + performanceBonus + "元");
        System.out.println("工龄补贴: " + (yearsOfService * 1000) + "元 (" + yearsOfService + "年)");
        System.out.println("应发工资: " + calculateSalary() + "元");
        System.out.println("===================");
    }

    // Getter方法
    public int getYearsOfService() {
        return yearsOfService;
    }

    public double getPerformanceBonus() {
        return performanceBonus;
    }
}

// 3. 经理类 Manager
class Manager extends Employee implements SalaryCalculator {
    private int teamSize;
    private double managementBonus;

    public Manager(String employeeId, String name, String department,
                   double baseSalary, int teamSize, double managementBonus) {
        super(employeeId, name, department, baseSalary);
        this.teamSize = teamSize;
        this.managementBonus = managementBonus;
    }

    @Override
    public void work() {
        System.out.println("正在管理团队和制定计划");
    }

    @Override
    public double calculateSalary() {
        // 总工资 = 基本工资 + 管理津贴 + (团队人数 * 500)
        return baseSalary + managementBonus + (teamSize * 500);
    }

    @Override
    public void displaySalaryDetails() {
        System.out.println("===== 工资明细 =====");
        System.out.println("基本工资: " + baseSalary + "元");
        System.out.println("管理津贴: " + managementBonus + "元");
        System.out.println("团队管理补贴: " + (teamSize * 500) + "元 (团队" + teamSize + "人)");
        System.out.println("应发工资: " + calculateSalary() + "元");
        System.out.println("===================");
    }

    // Getter方法
    public int getTeamSize() {
        return teamSize;
    }

    public double getManagementBonus() {
        return managementBonus;
    }
}

// 4. 主程序
public class EmployeeManagementSystem {
    public static void main(String[] args) {
        System.out.println("========== 员工管理系统 ==========\n");

        // 创建员工对象
        RegularEmployee emp1 = new RegularEmployee("E001", "张三", "技术部", 8000, 3, 2000);
        RegularEmployee emp2 = new RegularEmployee("E002", "李四", "市场部", 7000, 1, 1500);
        Manager manager1 = new Manager("M001", "王经理", "技术部", 12000, 5, 3000);

        // 将所有员工存储在一个数组中
        Employee[] employees = {emp1, emp2, manager1};

        double totalSalary = 0;

        // 遍历数组
        for (int i = 0; i < employees.length; i++) {
            System.out.println("【员工" + (i+1) + "】");
            employees[i].displayInfo();
            System.out.print("工作状态: ");
            employees[i].work();

            // 计算工资
            if (employees[i] instanceof SalaryCalculator) {
                SalaryCalculator calculator = (SalaryCalculator) employees[i];
                System.out.println("月工资: " + calculator.calculateSalary() + "元");
                calculator.displaySalaryDetails();
                totalSalary += calculator.calculateSalary();
            }

            System.out.println(); // 空行分隔
        }

        // 显示所有员工的总工资
        System.out.println("========== 工资汇总 ==========");
        System.out.println("员工总数: " + employees.length + "人");
        System.out.println("总工资支出: " + totalSalary + "元");
        System.out.println("平均工资: " + (totalSalary / employees.length) + "元");

        // 添加更多示例展示类的功能
        System.out.println("\n========== 更多示例 ==========");

        // 创建更多员工
        Manager manager2 = new Manager("M002", "刘总监", "市场部", 15000, 8, 5000);
        RegularEmployee emp3 = new RegularEmployee("E003", "赵五", "财务部", 7500, 5, 1800);

        Employee[] allEmployees = {emp1, emp2, manager1, manager2, emp3};

        System.out.println("公司所有员工列表:");
        for (Employee emp : allEmployees) {
            System.out.println(emp.getEmployeeId() + " - " + emp.getName() +
                    " (" + emp.getDepartment() + ") - " +
                    (emp instanceof Manager ? "经理" : "普通员工"));
        }
    }
}