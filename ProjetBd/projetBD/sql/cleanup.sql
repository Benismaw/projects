-- =====================================================
-- NETTOYAGE COMPLET DE LA BASE (RESET)
-- Respecte les contraintes de clé étrangère
-- =====================================================

-- 1. Tables dépendantes des commandes
DELETE FROM LigneCommande;
DELETE FROM InformationLivraison;
DELETE FROM PasseCommande;

-- 2. Livraison / adresses
DELETE FROM AdresseLivraison;

-- 3. Lots, pertes
DELETE FROM Lot;
DELETE FROM Pertes;

-- 4. Conditionnement et Contenant
DELETE FROM Conditionnement;
DELETE FROM Contenant;
DELETE FROM ArticleEnVente;

-- 5. Disponibilité des produits
DELETE FROM Disponibilite;

-- 6. Commandes
DELETE FROM Commande;

-- 7. Relations du producteur
DELETE FROM EstDoType;

-- 8. Produits et producteurs
DELETE FROM Produit;
DELETE FROM Producteur;

-- 9. Clients
DELETE FROM Client;

-- 10. Types d'activité
DELETE FROM TypeActivite;

-- Finaliser
COMMIT;
