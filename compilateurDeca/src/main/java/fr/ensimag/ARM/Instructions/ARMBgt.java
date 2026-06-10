package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMBgt extends ARMInstruction{
    private String label;

    public ARMBgt(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "bgt " + label;
    }
}
