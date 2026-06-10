//role :menu pour tester toutes les fonctionnalités.
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        boolean running = true;

        while(running) {
            System.out.println("1. Afficher catalogue");
            System.out.println("2. Afficher clients");
            System.out.println("3. Ajouter client");
            System.out.println("4. Afficher stocks");
            System.out.println("5. Ajouter lot");
            System.out.println("6. Supprimer lot");
            System.out.println("7. Tester alertes péremption");//et les autres transactions aussi
            System.out.println("8. Quitter");

            int choice = sc.nextInt();
            sc.nextLine(); // consommer retour chariot

            switch(choice) {
                case 1 -> Catalog.afficherProduits();
                case 2 -> ClientUtils.afficherClients();
                case 3 -> {
                    System.out.print("Email: "); String email = sc.nextLine();
                    System.out.print("Nom: "); String nom = sc.nextLine();
                    System.out.print("Prénom: "); String prenom = sc.nextLine();
                    ClientUtils.ajouterClient(email, nom, prenom);
                }
                case 4 -> StockUtils.afficherStocks();
                case 5 -> System.out.println("À compléter avec saisie et appel StockUtils.ajouterLot");
                case 6 -> System.out.println("À compléter avec saisie et appel StockUtils.supprimerLot");
                case 7 -> new GérerAlertesPeremption();
                case 8 -> running = false;
                default -> System.out.println("Choix invalide.");
            }
        }

        try { DBConnection.closeConnection(); } catch (Exception e) {}
    }
}
