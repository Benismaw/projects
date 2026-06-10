package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMBeq extends ARMInstruction {

    private String label;

    public ARMBeq(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "beq " + label;
    }
}