import java.sql.*;

public class CloturerCommande {

    static final String CONN_URL = "jdbc:oracle:thin:@oracle1.ensimag.fr:1521:oracle1";
    static final String USER = "scott";
    static final String PASSWD = "tiger";

    public static void main(String[] args) {

        if (args.length != 1) {
            System.err.println("Usage : java CloturerCommande <idCommande>");
            System.exit(1);
        }

        int idCommande = Integer.parseInt(args[0]);

        try {
            // Chargement driver Oracle
            DriverManager.registerDriver(new oracle.jdbc.driver.OracleDriver());

            // Connexion
            Connection conn = DriverManager.getConnection(CONN_URL, USER, PASSWD);
            conn.setAutoCommit(false);   // On gère la transaction nous-mêmes

            // 1 — Récupérer infos de la commande (dont STATUT)
            String sqlInfos = """
                SELECT modeRecuperation, modePaiement, statut
                FROM Commande
                WHERE idCommande = ?
            """;

            PreparedStatement infoStmt = conn.prepareStatement(sqlInfos);
            infoStmt.setInt(1, idCommande);

            ResultSet rs = infoStmt.executeQuery();

            if (!rs.next()) {
                System.out.println("La commande " + idCommande + " n'existe pas.");
                conn.close();
                return;
            }

            String modeRecuperation = rs.getString("modeRecuperation");
            String modePaiement = rs.getString("modePaiement");
            String statutCommande = rs.getString("statut");

            // -------------------------------------------------------------
            // Vérifier que la commande est PRETE AVANT gestion du stock
            // -------------------------------------------------------------
            if (!statutCommande.equalsIgnoreCase("PRETE")) {
                System.out.println(" La commande n'est pas PRETE → pas de gestion de stock.");
                conn.close();
                return;
            }

            System.out.println("La commande est PRETE → on gère le stock.");

            // -------------------------------------------------------------
            // Calcul montant
            // -------------------------------------------------------------
            String sqlMontant =
                    "SELECT SUM(quantite * prixUnitaire) AS total " +
                    "FROM ligneCommande WHERE idCommande = ?";

            PreparedStatement montantStmt = conn.prepareStatement(sqlMontant);
            montantStmt.setInt(1, idCommande);
            ResultSet mrs = montantStmt.executeQuery();
            mrs.next();
            double montant = mrs.getDouble("total");

            // -------------------------------------------------------------
            // 2 — Gestion du stock 
            // -------------------------------------------------------------
            System.out.println("\n--- Mise à jour du stock ---");

            PreparedStatement lignesSt = conn.prepareStatement(
                "SELECT idArticle, quantite FROM ligneCommande WHERE idCommande = ?"
            );
            lignesSt.setInt(1, idCommande);
            ResultSet lignes = lignesSt.executeQuery();

            while (lignes.next()) {

                int idArticle = lignes.getInt("idArticle");
                double qteCommandee = lignes.getDouble("quantite");
                PreparedStatement typeStmt = conn.prepareStatement(
                    "SELECT P.type FROM Produit P " +
                    "JOIN conditionnement C ON P.idProduit = C.idProduit " +
                    "WHERE C.idArticle = ?"
                );
                typeStmt.setInt(1, idArticle);
                ResultSet rsType = typeStmt.executeQuery();
                if (rsType.next() && rsType.getString("type").equalsIgnoreCase("commande")) {
                    System.out.println("Produit " + idArticle + " est sur commande → pas de gestion FIFO.");
                    continue;
                }

                System.out.println("Article " + idArticle +
                                " : quantité commandée = " + qteCommandee);

                PreparedStatement lotsSt = conn.prepareStatement(
                    "SELECT idLot, quantite FROM lot " +
                    "WHERE idArticle = ? AND quantite > 0 " +
                    "ORDER BY datePeremption ASC"
                );
                lotsSt.setInt(1, idArticle);
                ResultSet lots = lotsSt.executeQuery();

                while (lots.next() && qteCommandee > 0) {

                    int idLot = lots.getInt("idLot");
                    double dispo = lots.getDouble("quantite");

                    double retirer = Math.min(dispo, qteCommandee);

                    PreparedStatement updLot = conn.prepareStatement(
                        "UPDATE lot SET quantite = quantite - ? " +
                        "WHERE idArticle = ? AND idLot = ?"
                    );
                    updLot.setDouble(1, retirer);
                    updLot.setInt(2, idArticle);
                    updLot.setInt(3, idLot);
                    updLot.executeUpdate();

                    System.out.println("  Lot " + idLot +
                                    " → - " + retirer + " unités");

                    qteCommandee -= retirer;
                }

                if (qteCommandee > 0) {
                    System.err.println("Stock insuffisant pour l'article " + idArticle);
                    conn.rollback();
                    return;
                }
            }

            System.out.println("Stock mis à jour.");

            // -------------------------------------------------------------
            // 3 — Mise à jour du statut FINAL
            // -------------------------------------------------------------
            String nouveauStatut = modeRecuperation.equalsIgnoreCase("LIVRAISON") ? "LIVREE" : "RECUPEREE";

            String sqlUpdate = "UPDATE Commande SET statut = ? WHERE idCommande = ?";
            PreparedStatement stmtUpdate = conn.prepareStatement(sqlUpdate);
            stmtUpdate.setString(1, nouveauStatut);
            stmtUpdate.setInt(2, idCommande);
            stmtUpdate.executeUpdate();

            System.out.println("Statut mis à jour → " + nouveauStatut);

            // --- Date de récupération si en boutique ---
            if (modeRecuperation.equalsIgnoreCase("EN BOUTIQUE")) {
                PreparedStatement datRec = conn.prepareStatement(
                    "UPDATE Commande SET dateRecuperation = SYSDATE WHERE idCommande = ?"
                );
                datRec.setInt(1, idCommande);
                datRec.executeUpdate();
                System.out.println("Date de récupération enregistrée.");
            }
            
            // 4 — Paiement si paiement "EN BOUTIQUE"
            if (modePaiement.equalsIgnoreCase("EN BOUTIQUE")) {
                PreparedStatement payStmt = conn.prepareStatement(
                    "UPDATE Commande SET commandePayee = 1, datePaiement = SYSDATE WHERE idCommande = ?"
                );
                payStmt.setInt(1, idCommande);
                payStmt.executeUpdate();

                System.out.println("Paiement enregistré (montant = " + montant + "€)");
            }

            // 5 — Mise à jour livraison
            if (modeRecuperation.equalsIgnoreCase("LIVRAISON")) {
                double frais = montant * 0.10;

                PreparedStatement livStmt = conn.prepareStatement(
                    "UPDATE InfoLiv SET frais = ?, dateLiv = SYSDATE WHERE idCommande = ?"
                );

                livStmt.setDouble(1, frais);
                livStmt.setInt(2, idCommande);
                livStmt.executeUpdate();

                System.out.println("Livraison mise à jour → frais = " + frais + " €");
            }

            // 6 — Commit final
            conn.commit();
            System.out.println("\nCommande " + idCommande + " clôturée avec succès ");

            conn.close();

        } catch (SQLException e) {
            System.err.println("Erreur SQL : " + e.getMessage());
            e.printStackTrace();
        }
    }
}
