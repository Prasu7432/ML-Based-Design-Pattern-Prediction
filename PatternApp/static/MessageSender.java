// MessageSender.java
public class MessageSender {
    public void sendMessage(String type, String message) {
        if (type.equalsIgnoreCase("Email")) {
            System.out.println("Sending Email: " + message);
        } else if (type.equalsIgnoreCase("SMS")) {
            System.out.println("Sending SMS: " + message);
        } else if (type.equalsIgnoreCase("Push")) {
            System.out.println("Sending Push Notification: " + message);
        } else {
            System.out.println("Invalid message type!");
        }
    }
}
// Main.java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter message type (Email, SMS, Push):");
        String type = scanner.nextLine();

        System.out.println("Enter your message:");
        String message = scanner.nextLine();

        MessageSender sender = new MessageSender();
        sender.sendMessage(type, message);

        scanner.close();
    }
}