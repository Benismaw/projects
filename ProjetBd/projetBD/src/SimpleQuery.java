import java.sql.*;
import java.util.Scanner;

public class SimpleQuery {

    static final String CONN_URL = "jdbc:oracle:thin:@oracle1.ensimag.fr:1521:oracle1";
    static final String USER = "elharchr";
    static final String PASSWD = "elharchr";

    private Connection conn;

    public SimpleQuery() {
        try {
            // Charger le driver Oracle
            System.out.print("Loading Oracle driver... ");
            Class.forName("oracle.jdbc.driver.OracleDriver");
            System.out.println("loaded.");

            // Connexion
            System.out.print("Connecting to the database... ");
            conn = DriverManager.getConnection(CONN_URL, USER, PASSWD);
            System.out.println("connected.");

            conn.setAutoCommit(false);

            Scanner scanner = new Scanner(System.in);
            String sql;

            System.out.println("\n=== MODE SQL INTERACTIF ===");
            System.out.println("Tapez une requête SQL ou 'exit' pour quitter.");
            System.out.println("--------------------------------------------");

            while (true) {
                System.out.print("\nSQL> ");
                sql = scanner.nextLine().trim();

                if (sql.equalsIgnoreCase("exit")) break;
                if (sql.isEmpty()) continue;

                try {
                    // Si c’est un SELECT → executeQuery
                    if (sql.toLowerCase().startsWith("select")) {
                        PreparedStatement stmt = conn.prepareStatement(sql);
                        ResultSet rs = stmt.executeQuery();
                        dumpResultSet(rs);
                        rs.close();
                        stmt.close();
                    }
                    // Sinon → executeUpdate
                    else {
                        PreparedStatement stmt = conn.prepareStatement(sql);
                        int rows = stmt.executeUpdate();
                        System.out.println("OK (" + rows + " lignes affectées)");
                        stmt.close();
                    }

                    conn.commit();

                } catch (SQLException e) {
                    System.err.println("Erreur SQL : " + e.getMessage());
                    conn.rollback();
                }
            }

            conn.close();
            System.out.println("Connexion fermée.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void dumpResultSet(ResultSet rset) throws SQLException {
        ResultSetMetaData meta = rset.getMetaData();
        int columns = meta.getColumnCount();

        // Affichage des en-têtes
        for (int i = 1; i <= columns; i++) {
            System.out.print(meta.getColumnName(i) + "\t");
        }
        System.out.println("\n--------------------------------------------------");

        // Affichage des lignes
        while (rset.next()) {
            for (int i = 1; i <= columns; i++) {
                System.out.print(rset.getString(i) + "\t");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        new SimpleQuery();
    }
}
