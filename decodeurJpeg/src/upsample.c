#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mcu.h"

// On effectue l'up-sampling d'un bloc Cb ou Cr déja sous échantillonné
void upsample_comp(bloc comp[], int fact_h, int fact_v,uint8_t largeur,uint8_t hauteur) {
    int num_blocs_h =largeur/(8*fact_h);
    int num_blocs_v =hauteur/(8*fact_v);

    // nouveau tableau temporaire pour stocker le résultat du up-sampling 
    uint8_t nouv_cb[largeur*hauteur];

    // Parcours des blocs dans comp[]
    for (int bloc_v = 0; bloc_v < num_blocs_v; bloc_v++) {
        for (int bloc_h = 0; bloc_h < num_blocs_h; bloc_h++) {
            uint8_t (*bloc)[8][8]=&comp[bloc_v*num_blocs_h+bloc_h];

            // on duplique dans le tableau temporaire
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    uint8_t val =(*bloc)[i][j];

                    // On place la valeur dans toutes les positions étendues correspondantes
                    for (int x = 0; x < fact_v; x++) {
                        for (int y = 0; y < fact_h; y++) {
                            int nouv_i= bloc_v*8*fact_v+i*fact_v+x;
                            int nouv_j= bloc_h*8*fact_h+j*fact_h+y;
                            nouv_cb[nouv_i*largeur+nouv_j] =val;
                        }
                    }
                }
            }
        }
    }

    // Copie du contenu de nouv_cb dans les blocs de comp[]
    int index = 0;
    for (int i = 0; i < hauteur; i += 8) {
        for (int j = 0; j < largeur; j += 8) {
            uint8_t (*bloc)[8][8]=&comp[index];
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    (*bloc)[x][y] = nouv_cb[(i+x)*largeur+(j+y)];
                }
            }
            index++;
        }
    }
}

// Fonction principale qui applique le bon type d'up-sampling selon le format de sous-échantillonnage
void upsampling(MCU *mcu ,uint8_t h1,uint8_t v1,uint8_t h2,uint8_t v2,uint8_t h3,uint8_t v3){
    uint8_t largeur_mcu=8*h1;
    uint8_t hauteur_mcu=8*v1;

    // Par sécurité, on vérifie que le pointeur n’est pas NULL
    if (mcu==NULL){
        return ;
    }
    // verifier si il s'agit d'un cas de sous echantillonage pour Cb
    // Si sous echantillonage horizontal 4:2:2
    if (h2*2==h1 && v2==v1){
        upsample_comp(mcu->Cb,2,1,largeur_mcu,hauteur_mcu);
    }
    // Si sous echantillonage horizontal et vertical  4:2:0
    else if (h2*2==h1 && v2*2==v1){
        upsample_comp(mcu->Cb,2,2,largeur_mcu,hauteur_mcu);
    }
    // Si sous echantillonage vertical 4:2:2
    else if (h2==h1 && v2*2==v1){
        upsample_comp(mcu->Cb,1,2,largeur_mcu,hauteur_mcu);
    }


    // verifier si il s'agit d'un cas de sous echantillonage pour Cr
    // Si sous echantillonage horizontal 4:2:2
    if (h3*2==h1 && v3==v1){
        upsample_comp(mcu->Cr,2,1,largeur_mcu,hauteur_mcu);
    }
    // Si sous echantillonage horizontal et vertical  4:2:0
    else if (h3*2==h1 && v3*2==v1){
        upsample_comp(mcu->Cr,2,2,largeur_mcu,hauteur_mcu);
    }
    // Si sous echantillonage vertical 4:2:2
    else if (h3==h1 && v3*2==v1){
        upsample_comp(mcu->Cr,1,2,largeur_mcu,hauteur_mcu);
    }

}
