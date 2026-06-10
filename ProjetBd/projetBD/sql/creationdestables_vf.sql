create table articleEnVente (
            idArticle integer primary key );
            
create table pertes (
            idArticle integer not null references articleEnVente(idArticle),
            idPerte integer not null,
            datePerte date not null,
            nature varchar(50) not null check(nature in ('vol', 'casse', 'abimé')),
            quantité float not null,
            primary key(idArticle, idPerte) );
            
create table contenants (
            idArticle integer primary key references articleEnVente(idArticle),
            prix float not null,
            typeContenant varchar(50) not null,
            stockDisponible integer not null,
            capacite float not null,
            caractere varchar(30) not null check(caractere in ('réutilisable', 'jetable') ));
            
create table conditionnement (
            idArticle integer primary key references articleEnVente(idArticle),
            prixAchat float not null,
            prixVente float not null,
            poids float,
            typeConditionnement varchar(30) check(typeConditionnement in ('vrac', 'pré-conditionné')),
            idProduit integer references Produit(idProduit) );
        
create table lot (
            idArticle integer not null references conditionnement(idArticle),
            idLot integer not null, 
            quantite float not null, 
            dateReception date not null,
            datePeremption date not null, 
            primary key(idArticle, idLot) );
            
======================================================
-- Producteur
-- ======================================================
CREATE TABLE Producteur (
    idProducteur INT PRIMARY KEY,
    nom VARCHAR2(30) NOT NULL,
    mail VARCHAR2(100) NOT NULL,
    CONSTRAINT chk_mail_prod CHECK(mail LIKE '%@%.%'),
    numTel VARCHAR2(15) NOT NULL,
    numFixe VARCHAR2(15) NOT NULL,
    pays VARCHAR2(50) NOT NULL,
    ville VARCHAR2(50) NOT NULL,
    quartier VARCHAR2(100) NOT NULL,
    latitude NUMBER(9,6) NOT NULL,
    longitude NUMBER(9,6) NOT NULL,
    CHECK (latitude BETWEEN -90 AND 90),
    CHECK (longitude BETWEEN -180 AND 180)
);

-- ======================================================
-- typeActivite
-- ======================================================
CREATE TABLE typeActivite (
    typeAc VARCHAR2(50) PRIMARY KEY
);

-- ======================================================
-- estDeType association
-- ======================================================
CREATE TABLE estDeType (
    idProducteur INT,
    typeAc VARCHAR2(50) NOT NULL,
    PRIMARY KEY(idProducteur, typeAc),
    FOREIGN KEY (idProducteur) REFERENCES Producteur(idProducteur),
    FOREIGN KEY (typeAc) REFERENCES typeActivite(typeAc)
);

-- ======================================================
-- Client
-- ======================================================
CREATE TABLE Client (
    email VARCHAR2(100) PRIMARY KEY,
    CONSTRAINT chk_mail_client CHECK(email LIKE '%@%.%'),
    nom VARCHAR2(30) NOT NULL,
    prenom VARCHAR2(30) NOT NULL,
    tel VARCHAR2(15) NOT NULL
);

-- ======================================================
-- Adresse Livraison
-- ======================================================
CREATE TABLE adresseLivraison (
    idAdresse INT PRIMARY KEY,
    email VARCHAR2(100) NOT NULL,
    pays VARCHAR2(50) NOT NULL,
    ville VARCHAR2(50) NOT NULL,
    quartier VARCHAR2(100) NOT NULL,
    numero VARCHAR2(15) NOT NULL,
    FOREIGN KEY (email) REFERENCES Client(email)
);

-- ======================================================
-- Commande
-- ======================================================
CREATE TABLE Commande (
    idCommande INT PRIMARY KEY,
    dateCommande DATE NOT NULL,
    heure VARCHAR2(5) NOT NULL,   
    statut VARCHAR2(20) CHECK (statut IN
        ('EN PREPARATION', 'PRETE','EN LIVRAISON','RECUPERÉE/LIVRÉE','ANULLÉE')),
    modePaiement VARCHAR2(20) CHECK (modePaiement IN ('EN LIGNE', 'EN BOUTIQUE')),
    modeRecuperation VARCHAR2(20) CHECK (modeRecuperation IN ('EN BOUTIQUE', 'À DOMICILE'))
);



-- ======================================================
-- InfoLiv
-- ======================================================
CREATE TABLE InfoLiv (
idCommande int REFERENCES Commande(idCommande),
    idLivraison INT ,
    idAdresse INT REFERENCES adresseLivraison(idAdresse),
    frais NUMBER NOT NULL,
    dateLiv DATE NOT NULL,
    primary key(idCommande,idLivraison)

);

-- ======================================================
-- passeCommande : association Client - Commande
-- ======================================================
CREATE TABLE passeCommande (
    idCommande INT,
    email VARCHAR2(100) NOT NULL,
    PRIMARY KEY(idCommande, email),
    FOREIGN KEY (idCommande) REFERENCES Commande(idCommande),
    FOREIGN KEY (email) REFERENCES Client(email)
);

-- ======================================================
-- LigneCommande
-- ======================================================
CREATE TABLE ligneCommande (
    idCommande INT NOT NULL,
    idLigne INT NOT NULL,
    idArticle INT NOT NULL,
    quantite FLOAT NOT NULL,
    prixUnitaire FLOAT NOT NULL,
    PRIMARY KEY(idCommande, idLigne),
    FOREIGN KEY(idCommande) REFERENCES Commande(idCommande),
    FOREIGN KEY(idArticle) REFERENCES ArticleEnVente(idArticle)
);

-- ======================================================
-- Produit
-- ======================================================
CREATE TABLE Produit (
    idProduit            INT PRIMARY KEY,
    nom                  VARCHAR(50) NOT NULL,
    categorie            VARCHAR(20) NOT NULL,
    description          VARCHAR(200),
    caracteristiques     VARCHAR(300),

    type                 VARCHAR(10) NOT NULL 
        CHECK (type IN ('stock','commande')),

    delaiDisponibilite   INT,
        CHECK (
        (type = 'commande' AND delaiDisponibilite IS NOT NULL AND delaiDisponibilite > 0)
        OR
        (type = 'stock' AND delaiDisponibilite IS NULL)
    ),

    idProducteur         INT NOT NULL,
    FOREIGN KEY (idProducteur) REFERENCES Producteur(idProducteur)
);
--table disponibilité :entité faible de produitstcok
CREATE TABLE Disponibilite (
    idProduit INT NOT NULL,
    debut     INT NOT NULL,
    fin       INT NOT NULL,

    PRIMARY KEY (idProduit, debut),

    FOREIGN KEY (idProduit) REFERENCES Produit(idProduit),

    CHECK (fin > debut)
);
--contraite pour empêcher une disponibilité pour un produit de type commande
ALTER TABLE Disponibilite
ADD CONSTRAINT ck_dispo_stock
CHECK (
    idProduit IN (SELECT idProduit FROM Produit WHERE type = 'stock')
);

drop table conditionnement;
drop table lignecommande;
drop table lot;
drop table produit;
drop table commande;
drop table passecommande;
drop table infoliv;
drop table pertes;
drop table contenants;
drop table articleenvente;
drop table disponibilite;
drop table produit;
