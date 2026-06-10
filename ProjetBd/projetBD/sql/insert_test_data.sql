-- Producteurs
INSERT INTO Producteur VALUES (1,'Fromagerie du Valais','contact@fromage.ch','027700001',NULL,'Suisse','Sion','Centre',46.2333,7.3333);
INSERT INTO Producteur VALUES (2,'Ferme des Champs','ferme@champs.ch','027700002',NULL,'Suisse','Martigny','Plaine',46.1040,7.0740);
INSERT INTO Producteur VALUES (3,'Apiculteur du Mont','apiculteur@mont.ch','027700003',NULL,'Suisse','Verbier','Hameau',46.0950,7.2230);
INSERT INTO Producteur VALUES (4,'Vigneron du Lac','vins@lac.ch','027700004',NULL,'Suisse','Neuchatel','Port',46.9920,6.9320);
INSERT INTO Producteur VALUES (5,'Huilerie Locale','huile@locale.ch','027700005',NULL,'Suisse','Geneve','Nord',46.2040,6.1430);

-- Types d’activité
INSERT INTO TypeActivite VALUES ('Fromage');
INSERT INTO TypeActivite VALUES ('Miel');
INSERT INTO TypeActivite VALUES ('Vin');
insert into TypeActivite values ('Huilerie');

-- EstDeType
INSERT INTO EstDeType VALUES (1,'Fromage');
INSERT INTO EstDeType VALUES (2,'Fromage');
INSERT INTO EstDeType VALUES (3,'Miel');
INSERT INTO EstDeType VALUES (4,'Vin');
INSERT INTO EstDeType VALUES (5,'Huilerie');

-- Clients
INSERT INTO Client VALUES ('alice@mail.com','Dupont','Alice','060000001');
INSERT INTO Client VALUES ('bob@mail.com','Martin','Bob','060000002');
INSERT INTO Client VALUES ('claire@mail.com','Nguyen','Claire','060000003');
INSERT INTO Client VALUES ('david@mail.com','Bernard','David','060000004');
INSERT INTO Client VALUES ('elena@mail.com','Rossi','Elena','060000005');
INSERT INTO Client VALUES ('fran@mail.com','Garcia','Fran','060000006');
INSERT INTO Client VALUES ('gul@mail.com','Khan','Gul','060000007');
INSERT INTO Client VALUES ('iris@mail.com','Haddad','Iris','060000008');
INSERT INTO Client VALUES ('jakob@mail.com','Olsen','Jakob','060000009');
INSERT INTO Client VALUES ('lena@mail.com','Moreau','Lena','060000010');

-- Adresses de livraison
INSERT INTO AdresseLivraison VALUES (1,'alice@mail.com','France','Paris','10e','12');
INSERT INTO AdresseLivraison VALUES (2,'alice@mail.com','France','Paris','11e','30');
INSERT INTO AdresseLivraison VALUES (3,'bob@mail.com','France','Lyon','3e','5');
INSERT INTO AdresseLivraison VALUES (4,'claire@mail.com','Suisse','Geneve','Centre','2');
INSERT INTO AdresseLivraison VALUES (5,'david@mail.com','France','Nice','Prom','7');
INSERT INTO AdresseLivraison VALUES (6,'elena@mail.com','France','Marseille','2e','12');
INSERT INTO AdresseLivraison VALUES (7,'fran@mail.com','Suisse','Zurich','Ouest','5');
INSERT INTO AdresseLivraison VALUES (8,'gul@mail.com','France','Lille','Centre','20');
INSERT INTO AdresseLivraison VALUES (9,'iris@mail.com','France','Bordeaux','Sud','18');
INSERT INTO AdresseLivraison VALUES (10,'jakob@mail.com','Suisse','Lausanne','Nord','3');

-- Articles en vente
INSERT INTO ArticleEnVente VALUES (0); INSERT INTO ArticleEnVente VALUES (1);
INSERT INTO ArticleEnVente VALUES (2); INSERT INTO ArticleEnVente VALUES (3);
INSERT INTO ArticleEnVente VALUES (4); INSERT INTO ArticleEnVente VALUES (5);
INSERT INTO ArticleEnVente VALUES (6); INSERT INTO ArticleEnVente VALUES (7);
INSERT INTO ArticleEnVente VALUES (8); INSERT INTO ArticleEnVente VALUES (9);

-- Contenants
INSERT INTO Contenant VALUES (0,2.50,'Bocal en verre',80,0.75,'REUTILISABLE');
INSERT INTO Contenant VALUES (1,0.20,'Sachet kraft',300,0.50,'JETABLE');
INSERT INTO Contenant VALUES (2,1.00,'Sac en tissu',120,1.00,'REUTILISABLE');
INSERT INTO Contenant VALUES (3,0.50,'Bocal plastique',150,0.50,'JETABLE');
INSERT INTO Contenant VALUES (4,3.00,'Bouteille verre',50,0.75,'REUTILISABLE');

-- Produits
INSERT INTO Produit VALUES (0,'Raclette','Fromage','Fromage à raclette','Local, bio','STOCK',NULL,1);
INSERT INTO Produit VALUES (1,'Gruyere','Fromage','Gruyere suisse','Local','STOCK',NULL,2);
INSERT INTO Produit VALUES (2,'Miel de Mont','Miel','Miel de montagne','Bio','STOCK',NULL,3);
INSERT INTO Produit VALUES (3,'Vin Blanc','Boisson','Vin blanc local','AOC','STOCK',NULL,4);
INSERT INTO Produit VALUES (4,'Truffe Blanche','Produit exception','Truffe blanche Piémont','Exception','COMMANDE',15,5);
INSERT INTO Produit VALUES (5,'Café Kopi Luwak','Boisson rare','Café indonésien','Rare','COMMANDE',20,5);
INSERT INTO Produit VALUES (6,'Vanille Bourbon','Épice','Vanille Bourbon premium','Gourmet','COMMANDE',15,5);
INSERT INTO Produit VALUES (7,'Safran Pur','Épice','Safran microdosé','Rare','COMMANDE',15,5);
INSERT INTO Produit VALUES (8,'Huile olive','Boisson','Huile extra vierge','Bio','STOCK',NULL,5);
INSERT INTO Produit VALUES (9,'Chocolat noir','Confiserie','Chocolat 70%','Bio','STOCK',NULL,5);

--  Conditionnements
INSERT INTO Conditionnement VALUES (0,24.0,34.0,NULL,'VRAC',0);
INSERT INTO Conditionnement VALUES (1,48.0,64.0,NULL,'VRAC',1);
INSERT INTO Conditionnement VALUES (2,4.0,7.5,250,'PRECONDITIONNE',2);
INSERT INTO Conditionnement VALUES (3,4.5,9.0,750,'PRECONDITIONNE',3);
INSERT INTO Conditionnement VALUES (4,4.0,6.0,NULL,'VRAC',4);
INSERT INTO Conditionnement VALUES (5,25.0,39.0,1000,'PRECONDITIONNE',5);
INSERT INTO Conditionnement VALUES (6,12.0,19.0,10,'PRECONDITIONNE',6);
INSERT INTO Conditionnement VALUES (7,6.0,12.0,NULL,'VRAC',7);
INSERT INTO Conditionnement VALUES (8,10.0,20.0,500,'PRECONDITIONNE',8);
INSERT INTO Conditionnement VALUES (9,2.5,5.0,NULL,'VRAC',9);

-- Lots (DLC/DLUO variés pour alerte)
INSERT INTO Lot VALUES (0,1,50,DATE '2025-11-20',DATE '2025-11-28','DLC');
INSERT INTO Lot VALUES (1,1,30,DATE '2025-11-21',DATE '2025-12-01','DLC');
INSERT INTO Lot VALUES (2,1,100,DATE '2025-11-01',DATE '2026-01-01','DLUO');
INSERT INTO Lot VALUES (3,1,20,DATE '2025-11-25',DATE '2025-12-02','DLC');
INSERT INTO Lot VALUES (4,1,15,DATE '2025-11-18',DATE '2025-11-27','DLC');

-- Pertes
INSERT INTO Perte VALUES (0,1,DATE '2025-11-18','CASSE',2);
INSERT INTO Perte VALUES (1,1,DATE '2025-11-19','VOL',1);
INSERT INTO Perte VALUES (2,1,DATE '2025-11-15','ABIME',3);

-- Commande
INSERT INTO Commande VALUES (1,DATE '2025-11-25','10:30','EN PREPARATION','EN LIGNE','RETRAIT',0,NULL,NULL);
INSERT INTO Commande VALUES (2,DATE '2025-11-24','11:00','PRETE','EN BOUTIQUE','RETRAIT',1,DATE '2025-11-24',DATE '2025-11-24');
INSERT INTO Commande VALUES (3,DATE '2025-11-23','12:00','EN LIVRAISON','EN LIGNE','LIVRAISON',0,NULL,NULL);
INSERT INTO Commande VALUES (4,DATE '2025-11-22','09:00','RECUPEREE','EN BOUTIQUE','RETRAIT',1,DATE '2025-11-22',DATE '2025-11-22');

-- PasseCommande
INSERT INTO PasseCommande VALUES (1,'alice@mail.com');
INSERT INTO PasseCommande VALUES (2,'bob@mail.com');
INSERT INTO PasseCommande VALUES (3,'claire@mail.com');
INSERT INTO PasseCommande VALUES (4,'david@mail.com');

-- LigneCommande
INSERT INTO LigneCommande VALUES (1,1,0,5,34.0);
INSERT INTO LigneCommande VALUES (1,2,1,3,64.0);
INSERT INTO LigneCommande VALUES (2,1,2,2,7.5);
INSERT INTO LigneCommande VALUES (3,1,3,1,9.0);
INSERT INTO LigneCommande VALUES (3,2,4,2,6.0);
INSERT INTO LigneCommande VALUES (4,1,0,1,34.0);

-- Disponibilite
INSERT INTO Disponibilite VALUES (0,DATE '2025-11-01',DATE '2025-12-31'); -- Raclette disponible
INSERT INTO Disponibilite VALUES (1,DATE '2025-11-01',DATE '2025-12-31'); -- Gruyere
INSERT INTO Disponibilite VALUES (2,DATE '2025-01-01',DATE '2025-06-30'); -- Miel (hors saison pour test)
INSERT INTO Disponibilite VALUES (3,DATE '2025-11-01',DATE '2025-12-31'); -- Vin

