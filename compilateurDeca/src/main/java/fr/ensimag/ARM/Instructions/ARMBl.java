package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMInstruction;

public class ARMBl extends ARMInstruction {
    
    private final String label;

    public ARMBl(String label) {
        this.label = label;
    }

    @Override 
    public String toString() {
        return "bl " + label;
    }
}
