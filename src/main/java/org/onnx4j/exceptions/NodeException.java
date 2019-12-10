package org.onnx4j.exceptions;

public class NodeException extends ForwarderException {

	private static final long serialVersionUID = -9110594428291367746L;

	private static final int ERR_CODE_BASE = 30000;

	public enum NodeExceptionEnums implements ExceptionEnums {

		OUTPUT_VALUE_INFO_UNDEFINED(ERR_CODE_BASE + 1, "Output of node named %s can not found value info"),
		UNSUPPORTED_ATTRIBUTE_TYPE(ERR_CODE_BASE + 1, "Unsupported attribute type named %s");

		public int code;
		public String message;

		private NodeExceptionEnums(int code, String message) {
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

	public NodeException(NodeExceptionEnums exceptionEnum) {
		this(exceptionEnum, new Object[] {});
	}

	public NodeException(NodeExceptionEnums exceptionEnum, Object... args) {
		super(String.format(exceptionEnum.getMessage(), args));
	}

}
