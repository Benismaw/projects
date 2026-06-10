-- ======================================================
-- CREATE TABLES
-- ======================================================

-- 1. Producteur
CREATE TABLE Producteur (
    idProducteur INT PRIMARY KEY,
    nomProducteur VARCHAR2(30) NOT NULL,
    mailProducteur VARCHAR2(100) NOT NULL ,
    CONSTRAINT cnt_mail_Producteur CHECK(mailProducteur LIKE '%@%.%'),
    numTel VARCHAR2(15) NOT NULL,
    numFixe VARCHAR2(20) NULL,
    pays VARCHAR2(50) NOT NULL,
    ville VARCHAR2(50) NOT NULL,
    quartier VARCHAR2(100) NOT NULL,
    latitude NUMBER(9,6) NOT NULL,
    CONSTRAINT cnt_latitude CHECK (latitude BETWEEN -90 AND 90),
    longitude NUMBER(9,6) NOT NULL ,
    CONSTRAINT cnt_longitude CHECK (longitude BETWEEN -180 AND 180)
);

-- 2. TypeActivite
CREATE TABLE TypeActivite (
    typeActivite VARCHAR2(50) PRIMARY KEY
);

-- 3. EstDeType (association)
CREATE TABLE EstDeType (
    idProducteur INT,
    typeActivite VARCHAR2(50) NOT NULL,
    PRIMARY KEY(idProducteur, typeActivite),
    FOREIGN KEY (idProducteur) REFERENCES Producteur(idProducteur),
    FOREIGN KEY (typeActivite) REFERENCES TypeActivite(typeActivite)
);

-- 4. Client
CREATE TABLE Client (
    mailClient  VARCHAR2(100) PRIMARY KEY ,
    CONSTRAINT cnt_mail_client CHECK(mailClient LIKE '%@%.%'),--contrainte cnt_mail_client
    nomClient VARCHAR2(30) NOT NULL,
    prenomClient VARCHAR2(30) NOT NULL,
    numTelClient VARCHAR2(15) NOT NULL
);

-- 5. AdresseLivraison
CREATE TABLE AdresseLivraison (
    idAdresse INT PRIMARY KEY,
    mailClient VARCHAR2(100) , -- nullable à cause de l'anonymat
    pays VARCHAR2(50) NOT NULL,
    ville VARCHAR2(50) NOT NULL,
    quartier VARCHAR2(100) NOT NULL,
    numero VARCHAR2(15),
    FOREIGN KEY (mailClient) REFERENCES Client(mailClient)
);

-- 6. ArticleEnVente
CREATE TABLE ArticleEnVente (
    idArticle INT PRIMARY KEY
);

-- 7. Pertes
CREATE TABLE Perte (
    idArticle INT NOT NULL REFERENCES ArticleEnVente(idArticle),
    idPerte INT NOT NULL,
    datePerte DATE NOT NULL,
    nature VARCHAR2(50) NOT NULL 
    CONSTRAINT cnt_nature_perte CHECK(nature IN ('VOL', 'CASSE', 'ABIME')),
    quantitePerdue FLOAT NOT NULL check(quantitePerdue>0),
    PRIMARY KEY(idArticle, idPerte)
);

-- 8. Contenants
CREATE TABLE Contenant (
    idArticle INT PRIMARY KEY REFERENCES ArticleEnVente(idArticle),
    prixContenant FLOAT NOT NULL check(prixContenant>0),
    typeContenant VARCHAR2(50) NOT NULL,
    stockContenant INT NOT NULL,
    capacite FLOAT NOT NULL check(capacite>0),
    caractere VARCHAR2(30) NOT NULL,
    CONSTRAINT cnt_caractere_contenant CHECK(caractere IN ('REUTILISABLE', 'JETABLE'))
);

-- 9. Produit
CREATE TABLE Produit (
    idProduit INT PRIMARY KEY,
    nomProduit VARCHAR2(50) NOT NULL,
    categorie VARCHAR2(50) NOT NULL,
    description VARCHAR2(200),
    caracteristiques VARCHAR2(300),
    typeProduit  VARCHAR2(10) NOT NULL CHECK(typeProduit IN ('STOCK','COMMANDE')),
    delaiDisponibilite INTEGER,
    CONSTRAINT cnt_delaidisponibilite_null CHECK (
        (typeProduit='COMMANDE' AND delaiDisponibilite IS NOT NULL AND delaiDisponibilite>0)
        OR 
        (typeProduit='STOCK' AND delaiDisponibilite IS NULL)
    ),
    idProducteur INT NOT NULL REFERENCES Producteur(idProducteur)
);

-- 10. Conditionnement
CREATE TABLE Conditionnement (
    idArticle INT PRIMARY KEY REFERENCES ArticleEnVente(idArticle),
    prixAchat FLOAT NOT NULL check(prixAchat>0),
    prixVente FLOAT NOT NULL check(prixVente>0),
    poids FLOAT,
    CONSTRAINT cnt_poids_null
    CHECK (
        (typeConditionnement = 'PRECONDITIONNE' AND poids IS NOT NULL AND poids > 0)
        OR
        (typeConditionnement = 'VRAC' AND poids IS NULL)
    ),
    typeConditionnement VARCHAR2(30) ,
    CONSTRAINT cnt_type_conditionnement CHECK(typeConditionnement IN ('VRAC','PRECONDITIONNE')),
    idProduit INT NOT NULL REFERENCES Produit(idProduit)
);

-- 11. Lot
CREATE TABLE Lot (
    idArticle INT NOT NULL REFERENCES Conditionnement(idArticle),
    idLot INT NOT NULL,
    quantiteDisponible FLOAT NOT NULL check (quantiteDisponible >0),
    dateReception DATE NOT NULL,
    datePeremption DATE NOT NULL,
    typePeremption VARCHAR2(5) NOT NULL,
    CONSTRAINT cnt_type_peremption CHECK(typePeremption IN ('DLC','DLUO')),
    PRIMARY KEY(idArticle, idLot)
);

-- 12. Sequence pour Commande
CREATE SEQUENCE seqCommande START WITH 1 INCREMENT BY 1;

-- 13. Commande
CREATE TABLE Commande (
    idCommande INT PRIMARY KEY,
    dateCommande DATE NOT NULL,
    heureCommande VARCHAR2(5) NOT NULL,

    statut VARCHAR2(20) NOT NULL,
    CONSTRAINT cnt_statut_com CHECK (
        statut IN ('EN PREPARATION','PRETE','EN LIVRAISON','RECUPEREE','LIVREE','ANNULEE')
    ),

    modePaiement VARCHAR2(20) NOT NULL,
    CONSTRAINT cnt_paiement CHECK (modePaiement IN ('EN LIGNE','EN BOUTIQUE')),

    modeRecuperation VARCHAR2(20) NOT NULL,
    CONSTRAINT cnt_recuperation CHECK (modeRecuperation IN ('RETRAIT','LIVRAISON')),

    commandePayee NUMBER(1) DEFAULT 0 CHECK (commandePayee IN (0,1)),
    dateRecuperation DATE,
    datePaiement DATE,

    CONSTRAINT cnt_paiement_effectue CHECK (
        (commandePayee = 0 AND datePaiement IS NULL)
        OR
        (commandePayee = 1 AND datePaiement IS NOT NULL)
    )
);

-- 14. InformationLivraison
CREATE TABLE InformationLivraison (
    idCommande INT NOT NULL REFERENCES Commande(idCommande),
    idLivraison INT NOT NULL,
    idAdresse INT NOT NULL REFERENCES AdresseLivraison(idAdresse),
    fraisLivraison float NOT NULL check(fraisLivraison>0),
    dateLivraison DATE,
    PRIMARY KEY(idCommande, idLivraison)
);

-- 15. PasseCommande
CREATE TABLE PasseCommande (
    idCommande INT not NULL,
    mailClient VARCHAR2(100) not NULL,
    PRIMARY KEY(idCommande,mailClient),
    FOREIGN KEY(idCommande) REFERENCES Commande(idCommande),
    FOREIGN KEY(mailClient) REFERENCES Client(mailClient)
);

-- 16. LigneCommande
CREATE TABLE LigneCommande (
    idCommande INT NOT NULL REFERENCES Commande(idCommande),
    idLigne INT NOT NULL,
    idArticle INT NOT NULL REFERENCES ArticleEnVente(idArticle),
    quantiteCommandee FLOAT NOT NULL check (quantiteCommandee >0),
    prixUnitaire FLOAT NOT NULL check (prixUnitaire>0),
    PRIMARY KEY(idCommande,idLigne)
);

-- 17. Disponibilite
CREATE TABLE Disponibilite (
    idProduit INT NOT NULL REFERENCES Produit(idProduit),
    debut DATE NOT NULL,
    fin DATE  NOT NULL,
    PRIMARY KEY(idProduit,debut),
    CHECK(fin>debut)
);