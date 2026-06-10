-- ======================================================
-- ALERTES DE PEREMPTION ET AJUSTEMENT DES PRIX
-- ======================================================

-- 1. Sélection des lots dont la date de péremption approche (7 jours)
SELECT l.idLot, l.idArticle, l.quantiteDisponible, p.nomProduit
FROM Lot l
JOIN Conditionnement c ON l.idArticle = c.idArticle
JOIN Produit p ON c.idProduit = p.idProduit
WHERE l.datePeremption BETWEEN SYSDATE AND SYSDATE + 7;

-- 2. Mise à jour du prix des articles concernés par une réduction
-- Ici, on applique la réduction uniquement aux articles dont au moins un lot arrive à péremption
UPDATE Conditionnement c
SET c.prixVente = c.prixVente * 0.7
WHERE c.idArticle IN (
    SELECT l.idArticle
    FROM Lot l
    WHERE l.datePeremption BETWEEN SYSDATE AND SYSDATE + 7
);
