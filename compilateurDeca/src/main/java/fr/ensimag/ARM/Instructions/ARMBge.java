package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMBge extends ARMInstruction{
    private String label;

    public ARMBge(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "bge " + label;
    }
}
