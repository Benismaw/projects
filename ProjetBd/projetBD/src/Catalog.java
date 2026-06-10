//role :afficher le catalogue des produits, leurs conditionnements et stocks.
import java.sql.*;

public class Catalog {
    public static void afficherProduits() {
        try (Connection conn = DBConnection.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(
                 "SELECT p.idProduit, p.nom, p.catégorie, c.typeConditionnement, c.prixVente " +
                 "FROM Produit p " +
                 "JOIN Conditionnement c ON p.idProduit = c.idProduit")) {

            System.out.println("=== Catalogue de Produits ===");
            while (rs.next()) {
                System.out.println(
                    "Produit: " + rs.getString("nom") +
                    " | Catégorie: " + rs.getString("catégorie") +
                    " | Conditionnement: " + rs.getString("typeConditionnement") +
                    " | Prix: " + rs.getDouble("prixVente")
                );
            }

        } catch (SQLException e) {
            System.err.println("Erreur affichage catalogue : " + e.getMessage());
        }
    }
}
