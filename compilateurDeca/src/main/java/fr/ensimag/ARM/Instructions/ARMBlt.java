package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMBlt extends ARMInstruction{
    private String label;

    public ARMBlt(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "blt " + label;
    }
}
