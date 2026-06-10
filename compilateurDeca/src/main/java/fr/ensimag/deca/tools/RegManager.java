package fr.ensimag.deca.tools;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;

/**
 * Gestionnaire des registres banalisés (R2 à R15).
 */
public class RegManager {
    private int maxRegUsed = 1; // R1 est le dernier registre scratch
    private int currentReg = 1;
    private final int nRegs;    // Limite fixée par l'option -r

    private int spilledCount = 0; // Compteur de registre sauvegarder sur la pile

    public RegManager(int nRegs) {
        this.nRegs = nRegs;
    }

    /**
     * Alloue un registre pour un calcul.
     * @return Le registre alloué (R2, R3...).
     */
    public GPRegister allocate(DecacCompiler compiler) {
        // Cas 1 : Il reste des registres physiques disponibles
        if (currentReg < nRegs - 1) {
            currentReg++;
            if (currentReg > maxRegUsed) {
                maxRegUsed = currentReg;
            }
            return Register.getR(currentReg);
        }

        // Cas 2 : Plus de registres -> SPILL (On déborde sur la pile)
        spilledCount++;

        // On récupère le dernier registre disponible
        GPRegister regMax = Register.getR(nRegs - 1);

        // On sauvegarde sa valeur actuelle sur la pile
        compiler.addInstruction(new PUSH(regMax));

        // On prévient le StackManager qu'on a empilé
        compiler.getStackManager().increment(1);

        // On retourne ce même registre
        return regMax;
    }

    public void free(DecacCompiler compiler) {
        // Cas 1 : On a des registres spillés -> On doit dépiler pour restaurer
        if (spilledCount > 0) {
            GPRegister regMax = Register.getR(nRegs - 1);

            // On restaure l'ancienne valeur
            compiler.addInstruction(new POP(regMax));

            // On prévient le StackManager qu'on a dépilé
            compiler.getStackManager().decrement(1);

            spilledCount--;
        }
        // Cas 2 : Pas de spill, on recule juste le compteur de registre
        else {
            if (currentReg > 1) {
                currentReg--;
            }
        }
    }

    public int getMaxRegUsed() {
        return maxRegUsed;
    }

    public void reset() {
        this.currentReg = 1;
        this.maxRegUsed = 1;
        this.spilledCount = 0;
    }
}