import java.sql.*;
import java.util.Scanner;

public class GérerAlertesPeremption {
  
  static final String CONN_URL = "jdbc:oracle:thin:@oracle1.ensimag.fr:1521:oracle1";
  static final String USER = "scott";     // A remplacer par notre compte oracle
  static final String PASSWD = "tiger";

  public GérerAlertesPeremption() {
    try {
      // Enregistrement du driver Oracle
      System.out.println("Loading Oracle thin driver...");
      DriverManager.registerDriver(new oracle.jdbc.driver.OracleDriver());
      System.out.println("Loaded.");

      // Etablissement de la connexion
      System.out.println("Connecting to the database...");
      Connection conn = DriverManager.getConnection(CONN_URL, USER, PASSWD);
      System.out.println("Connected.");

      // Démarrer la transaction (auto-commit désactivé)
      conn.setAutoCommit(false);
      conn.setTransactionIsolation(Connection.TRANSACTION_SERIALIZABLE);

      // Requête pour récupérer les alertes de péremption
      String sqlSelect = 
          "SELECT l.idLot, l.idArticle, l.quantite, p.nom " +
          "FROM lot l, Conditionnement c, Produit p " +
          "WHERE l.idArticle = c.idArticle " +
          "AND c.idProduit = p.idProduit " +
          "AND l.datePeremption BETWEEN SYSDATE AND SYSDATE + 7";
      
      PreparedStatement pstmt = conn.prepareStatement(sqlSelect);
      ResultSet rs = pstmt.executeQuery();

      System.out.println("=== ALERTES PÉREMPTION ===");

      boolean found = false;
      while (rs.next()) {
        found = true;
        System.out.println(
          "Lot " + rs.getInt("idLot") +
          " | Article " + rs.getInt("idArticle") +
          " | Quantité " + rs.getDouble("quantite") +
          " | Produit : " + rs.getString("nom")
        );
      }

      if (!found) {
        System.out.println("Aucune alerte dans les 7 prochains jours.");
        return;
      }

      // Demander à l'utilisateur s'il souhaite appliquer la réduction
      Scanner sc = new Scanner(System.in);
      System.out.println("\nAppliquer réduction automatique de 30% ? (o/n) : ");
      String rep = sc.nextLine();

      if (rep.equalsIgnoreCase("o")) {
        // Requête pour appliquer la réduction
        String sqlUpdate = 
            "UPDATE Conditionnement c " +
            "SET prixVente = prixVente * 0.7 " +
            "WHERE c.idArticle IN ( " +
            "SELECT l.idArticle " +
            "FROM lot l, Conditionnement c, Produit p " +
            "WHERE l.idArticle = c.idArticle " +
            "AND c.idProduit = p.idProduit " +
            "AND l.datePeremption BETWEEN SYSDATE AND SYSDATE + 7 )";

        int nb = pstmt.executeUpdate(sqlUpdate);
        conn.commit();  // Commit des modifications
        System.out.println("Réduction appliquée à " + nb + " articles.");
      } else {
        conn.rollback();  // Rollback si l'utilisateur refuse la réduction
        System.out.println("Aucune réduction appliquée.");
      }

      // Fermeture des ressources
      pstmt.close();
      conn.close();

    } catch (SQLException e) {
      System.err.println("Échec !");
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new GérerAlertesPeremption();  // Exécuter la méthode de gestion des alertes
  }
}