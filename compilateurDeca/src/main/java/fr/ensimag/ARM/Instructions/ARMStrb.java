package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMStrb extends ARMInstruction {
    
    private ARMRegister src;
    private ARMRegister addr;

    public ARMStrb(ARMRegister src, ARMRegister addr) {
        this.src = src;
        this.addr = addr;
    }

    @Override 
    public String toString() {
        return "strb " + src + ", [" + addr + "]";
    }
}
