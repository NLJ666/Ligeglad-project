import java.util.Scanner;
class Rectangle {
    double width;
    double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    public double area() {
        return width * height;
    }

    public double perimeter() {
        return 2 * (width + height);
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        System.out.print("请输入矩形的宽度: ");
        double width = in.nextDouble();

        System.out.print("请输入矩形的高度: ");
        double height = in.nextDouble();

        Rectangle rect = new Rectangle(width, height);

        System.out.println("矩形的面积为: " + rect.area());
        System.out.println("矩形的周长为: " + rect.perimeter());

        in.close();
    }
}