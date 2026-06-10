package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMPop extends ARMInstruction {

    private final ARMRegister reg;

    public ARMPop(ARMRegister reg) {
        this.reg = reg;
    }

    @Override
    public String toString() {
        return "pop {" + reg + "}";
    }    
}
