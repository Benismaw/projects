//rôle :gérer la saisie et affichage des clients pour tests.
import java.sql.*;

public class ClientUtils {
    public static void afficherClients() {
        try (Connection conn = DBConnection.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT email, nom, prenom FROM Client")) {

            System.out.println("=== Clients ===");
            while (rs.next()) {
                System.out.println(
                    rs.getString("prenom") + " " + rs.getString("nom") +
                    " | Email: " + rs.getString("email")
                );
            }

        } catch (SQLException e) {
            System.err.println("Erreur affichage clients : " + e.getMessage());
        }
    }
    //a verifier email ou admail
    public static void ajouterClient(String email, String nom, String prenom) {
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(
                 "INSERT INTO Client(email, nom, prenom) VALUES (?, ?, ?)")) {

            pstmt.setString(1, email);
            pstmt.setString(2, nom);
            pstmt.setString(3, prenom);
            pstmt.executeUpdate();
            conn.commit();
            System.out.println("Client ajouté : " + nom + " " + prenom);

        } catch (SQLException e) {
            System.err.println("Erreur ajout client : " + e.getMessage());
        }
    }
}
