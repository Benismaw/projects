package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMBle extends ARMInstruction{
    private String label;

    public ARMBle(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "ble " + label;
    }
}
