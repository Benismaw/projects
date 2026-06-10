package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMB extends ARMInstruction {
    
    private String label;

    public ARMB(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "b " + label;
    }
    
}
