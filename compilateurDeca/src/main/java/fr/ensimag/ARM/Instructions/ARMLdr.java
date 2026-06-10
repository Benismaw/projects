package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMBinaryInstructionDValToReg;
import fr.ensimag.ARM.ARMDVal;
import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMLabelOperand;

public class ARMLdr extends ARMBinaryInstructionDValToReg {
    
    public ARMLdr(ARMGPRegister dst, ARMDVal address) {
        super(dst, address);
    }

    /**
     * Constructeur utilitaire pour labels : ldr r0, =str_0
     */
    public ARMLdr(ARMGPRegister dst, String label) {
        // On crée l'objet ARMLabelOperand (qui doit implémenter ARMDVal)
        super(dst, new ARMLabelOperand(new ARMLabel(label)));
    }

    @Override
    public String toString() {
        return "ldr " + getOperand1() + ", " + getOperand2();
    }
}
