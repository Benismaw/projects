package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMLdrMem extends ARMInstruction {
    private ARMRegister dst;
    private ARMRegister base;
    private int offset;

    public ARMLdrMem(ARMRegister dst, ARMRegister base, int offset) {
        this.dst = dst;
        this.base = base;
        this.offset = offset;
    }

    @Override
    public String toString() {
        return "ldr " + dst + ", [" + base + ", #" + offset + "]";
    }
}
