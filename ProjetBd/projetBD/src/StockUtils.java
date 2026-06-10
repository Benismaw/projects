//role :gérer les lots et stocks pour tests.
import java.sql.*;

public class StockUtils {
    public static void afficherStocks() {
        try (Connection conn = DBConnection.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(
                 "SELECT idLot, idArticle, quantite, dateReception, datePeremption, type FROM Lot")) {

            System.out.println("=== Stocks ===");
            while (rs.next()) {
                System.out.println(
                    "Lot: " + rs.getInt("idLot") +
                    " | Article: " + rs.getInt("idArticle") +
                    " | Quantité: " + rs.getDouble("quantite") +
                    " | Réception: " + rs.getDate("dateReception") +
                    " | Péremption: " + rs.getDate("datePeremption") +
                    " | Type: " + rs.getString("type")
                );
            }

        } catch (SQLException e) {
            System.err.println("Erreur affichage stocks : " + e.getMessage());
        }
    }

    public static void ajouterLot(int idLot, int idArticle, double quantite, Date dateReception, Date datePeremption, String type) {
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(
                 "INSERT INTO Lot(idLot, idArticle, quantite, dateReception, datePeremption, type) VALUES (?, ?, ?, ?, ?, ?)")) {

            pstmt.setInt(1, idLot);
            pstmt.setInt(2, idArticle);
            pstmt.setDouble(3, quantite);
            pstmt.setDate(4, dateReception);
            pstmt.setDate(5, datePeremption);
            pstmt.setString(6, type);
            pstmt.executeUpdate();
            conn.commit();
            System.out.println("Lot ajouté: " + idLot);

        } catch (SQLException e) {
            System.err.println("Erreur ajout lot : " + e.getMessage());
        }
    }

    public static void supprimerLot(int idLot) {
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement pstmt = conn.prepareStatement("DELETE FROM Lot WHERE idLot = ?")) {

            pstmt.setInt(1, idLot);
            pstmt.executeUpdate();
            conn.commit();
            System.out.println("Lot supprimé: " + idLot);

        } catch (SQLException e) {
            System.err.println("Erreur suppression lot : " + e.getMessage());
        }
    }
}
