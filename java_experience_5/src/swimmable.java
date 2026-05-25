// Swimmable 接口
interface Swimmable {
    void swim();
}

// Flyable 接口
interface Flyable {
    void fly();
}

// Duck 类实现两个接口
class Duck implements Swimmable, Flyable {
    @Override
    public void swim() {
        System.out.println("I can swim");
    }

    @Override
    public void fly() {
        System.out.println("I can fly");
    }
}