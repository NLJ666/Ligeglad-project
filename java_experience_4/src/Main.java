import java.util.Scanner;

// 图形父类
class Shape {
    double side;  // 边长

    public Shape(double side) {
        this.side = side;
    }

    // 求面积方法，将在子类中重写
    public double getArea() {
        return 0;
    }
}

// 正三角形类
class EquilateralTriangle extends Shape {
    public EquilateralTriangle(double side) {
        super(side);
    }

    @Override
    public double getArea() {
        // 正三角形面积公式: √3/4 * a²
        return Math.sqrt(3) / 4 * side * side;
    }
}

// 圆形类
class Circle extends Shape {
    public Circle(double side) {
        super(side);  // side表示半径
    }

    @Override
    public double getArea() {
        // 圆面积公式: π * r²
        return Math.PI * side * side;
    }
}

// 正方形类
class Square extends Shape {
    public Square(double side) {
        super(side);
    }

    @Override
    public double getArea() {
        // 正方形面积公式: a²
        return side * side;
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("请输入图形名称和边长（输入exit退出）: ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("exit")) {
                break;
            }

            // 分割输入
            String[] parts = input.split("\\s+");

            if (parts.length != 2) {
                System.out.println("Can’t calculate");
                continue;
            }

            String shapeName = parts[0];
            double side;

            try {
                side = Double.parseDouble(parts[1]);
                if (side <= 0) {
                    System.out.println("Can’t calculate");
                    continue;
                }
            } catch (NumberFormatException e) {
                System.out.println("Can’t calculate");
                continue;
            }

            // 根据图形名称创建对应对象并计算面积
            Shape shape = null;

            switch (shapeName.toLowerCase()) {
                case "圆形":
                case "circle":
                    shape = new Circle(side);
                    break;
                case "正三角形":
                case "equilateraltriangle":
                case "triangle":
                    shape = new EquilateralTriangle(side);
                    break;
                case "正方形":
                case "square":
                    shape = new Square(side);
                    break;
                default:
                    System.out.println("Can’t calculate");
                    continue;
            }

            if (shape != null) {
                double area = shape.getArea();
                System.out.printf("面积为: %.2f%n", area);
            }
        }

        scanner.close();
    }
}