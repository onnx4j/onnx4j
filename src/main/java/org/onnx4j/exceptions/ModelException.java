package org.onnx4j.exceptions;

public class ModelException extends ForwarderException {

	private static final long serialVersionUID = -9110594428291367746L;
	
	private static final int ERR_CODE_BASE = 10000;

	public enum ModelExceptionEnums implements ExceptionEnums {

		IR_VER_UNSUPPORTED(ERR_CODE_BASE + 1, "Model's ir version(%s) is newer than supported(%s)");

		public int code;
		public String message;

		private ModelExceptionEnums(int code, String message) {
			this.code = code;
			this.message = message;
		}

		@Override
		public int getCode() {
			return code;
		}

		@Override
		public String getMessage() {
			return message;
		}

	}
	
	public ModelException(ModelExceptionEnums exceptionEnum) {
		this(exceptionEnum, new Object[] {});
	}
	
	public ModelException(ModelExceptionEnums exceptionEnum, Object... args) {
		super(String.format(exceptionEnum.getMessage(), args));
	}

}
