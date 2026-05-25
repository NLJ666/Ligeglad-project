// Rentable.java
public interface Rentable {
    double calculateRent(int days);
    boolean isAvailable();
    void rent(String customerName);
    void returnVehicle();
}
