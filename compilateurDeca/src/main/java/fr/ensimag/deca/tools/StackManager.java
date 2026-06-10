package fr.ensimag.deca.tools;

import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;

/**
 * Gère le comptage symbolique de la pile pour générer le TSTO correct.
 */
public class StackManager {
    private int currentStack = 0;
    private int maxStack = 0;
    // Index pour les variables globales commence à 1(GB)
    private int globalIndex = 0;

    /**
     * Signale un empilement de n mots (ex: PUSH, ADDSP)
     */
    public void increment(int n) {
        currentStack += n;
        if (currentStack > maxStack) {
            maxStack = currentStack;
        }
    }

    /**
     * Signale un dépilement de n mots (ex: POP, SUBSP)
     */
    public void decrement(int n) {
        currentStack -= n;
    }

    /**
     * Alloue une nouvelle adresse pour une variable globale et incrémente l'index.
     */
    public DAddr allocGlobalVar() {
        
        globalIndex++;
        DAddr addr = new RegisterOffset(globalIndex, Register.GB);
        return addr;
    }

    public int getMaxStack() {
        return maxStack;
    }

    /**
     * Remet à zéro pour commencer un nouveau bloc (méthode)
     */
    public void reset() {
        currentStack = 0;
        maxStack = 0;
    }
        /**
     * Retourne le nombre total de mots alloués dans la zone globale.
     */
    public int getNbGlobalVars() {
        return globalIndex;
    }
}